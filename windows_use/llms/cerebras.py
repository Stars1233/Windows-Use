from cerebras.cloud.sdk.types.chat.completion_create_params import (
    MessageSystemMessageRequestTyped, MessageUserMessageRequestTyped, MessageAssistantMessageRequestTyped,
    MessageSystemMessageRequestContentUnionMember1Typed, MessageAssistantMessageRequestContentUnionMember1Typed,
    MessageUserMessageRequestContentUnionMember1Typed
)
from cerebras.cloud.sdk.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchemaJsonSchemaTyped, ResponseFormatResponseFormatJsonSchemaTyped
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from cerebras.cloud.sdk import Cerebras, AsyncCerebras
from typing import Iterator, AsyncIterator, Literal
from dataclasses import dataclass
from pydantic import BaseModel
from httpx import Client
import json
import os

class ChatCerebras(BaseChatLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: Client | None = None,
        strict_response_validation: bool = False,
        warm_tcp_connection: bool = True
    ):
        self.model = model
        if not api_key and not os.getenv("CEREBRAS_API_KEY"):
            raise ValueError("CEREBRAS_API_KEY is not set")
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.strict_response_validation = strict_response_validation
        self.warm_tcp_connection = warm_tcp_connection

    @property
    def client(self) -> Cerebras:
        """Initialize synchronous Cerebras client"""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation,
            "warm_tcp_connection": self.warm_tcp_connection
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return Cerebras(**kwargs)

    @property
    def async_client(self) -> AsyncCerebras:
        """Initialize asynchronous Cerebras client"""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation,
            "warm_tcp_connection": self.warm_tcp_connection
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return AsyncCerebras(**kwargs)

    @property
    def provider(self) -> str:
        return "cerebras"
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def serialize_messages(self, messages: list[BaseMessage]) -> list:
        """Convert BaseMessage objects to Cerebras message format"""
        serialized = []
        for message in messages:
            if isinstance(message, SystemMessage):
                content = [MessageSystemMessageRequestContentUnionMember1Typed(type="text", text=message.content)]
                serialized.append(MessageSystemMessageRequestTyped(role="system", content=content))
            elif isinstance(message, HumanMessage):
                content = [MessageUserMessageRequestContentUnionMember1Typed(type="text", text=message.content)]
                serialized.append(MessageUserMessageRequestTyped(role="user", content=content))
            elif isinstance(message, AIMessage):
                serialized_msg = {"role": "assistant", "content": message.content}
                if message.thinking:
                    serialized_msg["reasoning"] = message.thinking
                serialized.append(serialized_msg)
            elif isinstance(message, ToolMessage):
                # Assistant Tool Call
                serialized_msg = {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": message.id,
                        "type": "function",
                        "function": {
                            "name": message.name,
                            "arguments": json.dumps(message.params)
                        }
                    }]
                }
                if message.thinking:
                    serialized_msg["reasoning"] = message.thinking
                serialized.append(serialized_msg)
                # Tool Result
                if message.content:
                    serialized.append({
                        "role": "tool",
                        "tool_call_id": message.id,
                        "content": str(message.content)
                    })
            elif isinstance(message, ImageMessage):
                raise ValueError("Image messages are not supported by Cerebras yet.")
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return serialized
    
    def _prepare_tools(self, tools: list[Tool]) -> list[dict] | None:
        """Convert tools to Cerebras format"""
        if not tools:
            return None
        
        return [{'type': 'function', 'function': tool.json_schema} for tool in tools]
    
    def _prepare_response_format(self, structured_output: BaseModel | None, json_mode: bool):
        """Prepare response format for structured output or JSON mode"""
        if structured_output:
            return ResponseFormatResponseFormatJsonSchemaTyped(
                json_schema=ResponseFormatResponseFormatJsonSchemaJsonSchemaTyped(
                    name=structured_output.__class__.__name__,
                    description="Model output structured as JSON schema",
                    schema=structured_output.model_json_schema()
                ),
                type="json_schema"
            )
        elif json_mode:
            return {"type": "json_object"}
        return None
    
    def _parse_response(self, message, structured_output: BaseModel | None = None) -> tuple[BaseMessage | BaseModel, str | None]:
        """Parse Cerebras response into content and thinking"""
        thinking = getattr(message, 'reasoning', None)
        
        if structured_output:
            try:
                content = structured_output.model_validate_json(message.content)
                thinking = None  # Structured output doesn't include thinking
            except Exception as e:
                raise ValueError(f"Failed to parse structured output: {e}")
        elif message.tool_calls:
            tool_call = message.tool_calls[0]
            content = ToolMessage(
                id=tool_call.id,
                name=tool_call.function.name,
                params=json.loads(tool_call.function.arguments),
                content=None,
                thinking=thinking
            )
        else:
            content = AIMessage(content=message.content, thinking=thinking)
        
        return content, thinking
    
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Synchronous invocation of Cerebras chat completion"""
        try:
            cerebras_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "extra_body": {
                    "reasoning_effort": self.reasoning_effort
                }
            }
            
            if cerebras_tools:
                kwargs["tools"] = cerebras_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            completion = self.client.chat.completions.create(**kwargs)
            
            message = completion.choices[0].message
            content, thinking = self._parse_response(message, structured_output)

            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=ChatLLMUsage(
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    total_tokens=completion.usage.total_tokens
                )
            )
        except Exception as e:
            raise RuntimeError(f"Cerebras API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of Cerebras chat completion"""
        try:
            cerebras_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "extra_body": {
                    "reasoning_effort": self.reasoning_effort
                }
            }
            
            if cerebras_tools:
                kwargs["tools"] = cerebras_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            completion = await self.async_client.chat.completions.create(**kwargs)
            
            message = completion.choices[0].message
            content, thinking = self._parse_response(message, structured_output)

            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=ChatLLMUsage(
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    total_tokens=completion.usage.total_tokens
                )
            )
        except Exception as e:
            raise RuntimeError(f"Cerebras API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of Cerebras chat completion"""
        try:
            cerebras_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": True,
                "extra_body": {
                    "reasoning_effort": self.reasoning_effort
                }
            }
            
            if cerebras_tools:
                kwargs["tools"] = cerebras_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = self.client.chat.completions.create(**kwargs)
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    yield ChatLLMResponse(thinking=delta.reasoning)
        except Exception as e:
            raise RuntimeError(f"Cerebras API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of Cerebras chat completion"""
        try:
            cerebras_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": True,
                "extra_body": {
                    "reasoning_effort": self.reasoning_effort
                }
            }
            
            if cerebras_tools:
                kwargs["tools"] = cerebras_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = await self.async_client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    yield ChatLLMResponse(thinking=delta.reasoning)
        except Exception as e:
            raise RuntimeError(f"Cerebras API streaming error: {str(e)}")

    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata"""
        # Cerebras model context windows
        context_windows = {
            "llama3.1-8b": 8192,
            "llama3.1-70b": 8192,
            "llama-3.3-70b": 8192,
        }
        
        # Default context window
        context_window = 8192
        
        # Try to find matching model
        for key, window in context_windows.items():
            if key in self.model.lower():
                context_window = window
                break
        
        return ModelMetadata(
            name=self.model,
            context_window=context_window,
            owned_by="cerebras"
        )