from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam
)
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from typing import Iterator, AsyncIterator, Literal
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from openai import OpenAI, AsyncOpenAI
from dataclasses import dataclass
from pydantic import BaseModel
from httpx import Client
import requests
import json
import os


class ChatOpenRouter(BaseChatLLM):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium",
        timeout: int | None = None,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: Client | None = None,
        strict_response_validation: bool = False
    ):
        self.model = model
        if not api_key and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url or 'https://openrouter.ai/api/v1'
        self.timeout = timeout
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.strict_response_validation = strict_response_validation

    @property
    def client(self) -> OpenAI:
        """Initialize synchronous OpenRouter client (OpenAI-compatible)"""
        kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation
        }
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return OpenAI(**kwargs)

    @property
    def async_client(self) -> AsyncOpenAI:
        """Initialize asynchronous OpenRouter client (OpenAI-compatible)"""
        kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation
        }
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return AsyncOpenAI(**kwargs)

    @property
    def provider(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self.model

    def serialize_messages(self, messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
        """Convert BaseMessage objects to OpenRouter message format"""
        serialized = []
        for message in messages:
            if isinstance(message, SystemMessage):
                content = [ChatCompletionContentPartTextParam(type="text", text=message.content)]
                serialized.append(ChatCompletionSystemMessageParam(role="system", content=content))
            elif isinstance(message, HumanMessage):
                content = [ChatCompletionContentPartTextParam(type="text", text=message.content)]
                serialized.append(ChatCompletionUserMessageParam(role="user", content=content))
            elif isinstance(message, AIMessage):
                serialized_msg = {"role": "assistant", "content": message.content}
                if message.thinking:
                    # OpenRouter models often use reasoning_content
                    serialized_msg["reasoning_content"] = message.thinking
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
                    serialized_msg["reasoning_content"] = message.thinking
                serialized.append(serialized_msg)
                # Tool Result
                if message.content:
                    serialized.append({
                        "role": "tool",
                        "tool_call_id": message.id,
                        "content": str(message.content)
                    })
            elif isinstance(message, ImageMessage):
                message.scale_images(scale=0.7)
                images = message.convert_images("base64")
                content = [ChatCompletionContentPartTextParam(type="text", text=message.content)]
                
                for image in images:
                    # Ensure proper data URL format
                    if not image.startswith('data:'):
                        image = f"data:{message.mime_type};base64,{image}"
                    
                    content.append(ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=image, detail="auto")
                    ))
                
                serialized.append(ChatCompletionUserMessageParam(role="user", content=content))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return serialized

    def _prepare_tools(self, tools: list[Tool]) -> list[dict] | None:
        """Convert tools to OpenRouter format"""
        if not tools:
            return None
        
        return [{'type': 'function', 'function': tool.json_schema} for tool in tools]

    def _prepare_response_format(self, structured_output: BaseModel | None, json_mode: bool):
        """Prepare response format for structured output or JSON mode"""
        if structured_output:
            return ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name=structured_output.__class__.__name__,
                    description="Model output structured as JSON schema",
                    schema=structured_output.model_json_schema()
                )
            )
        elif json_mode:
            return {"type": "json_object"}
        return None

    def _parse_response(self, message: ChatCompletionMessage, structured_output: BaseModel | None = None) -> tuple[BaseMessage | BaseModel, str | None]:
        """Parse OpenRouter response into content and thinking"""
        # OpenRouter often includes reasoning in reasoning_content
        thinking = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None)
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            content = ToolMessage(
                id=tool_call.id,
                name=tool_call.function.name,
                params=json.loads(tool_call.function.arguments),
                content=None,
                thinking=thinking
            )
        elif structured_output:
            try:
                content = structured_output.model_validate_json(message.content)
            except Exception as e:
                raise ValueError(f"Failed to parse structured output: {e}")
        else:
            content = AIMessage(content=message.content or "", thinking=thinking)
        
        return content, thinking

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Synchronous invocation of OpenRouter chat completion"""
        try:
            or_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "extra_body": {
                    "reasoning":{
                            "effort": self.reasoning_effort
                        }
                }
            }
            
            if or_tools:
                kwargs["tools"] = or_tools
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
            raise RuntimeError(f"OpenRouter API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of OpenRouter chat completion"""
        try:
            or_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "extra_body": {
                    "reasoning":{
                            "effort": self.reasoning_effort
                    }
                }
            }
            
            if or_tools:
                kwargs["tools"] = or_tools
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
            raise RuntimeError(f"OpenRouter API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of OpenRouter chat completion"""
        try:
            or_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": True,
                "extra_body": {
                    "reasoning":{
                            "effort": self.reasoning_effort
                        }
                }
            }
            
            if or_tools:
                kwargs["tools"] = or_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = self.client.chat.completions.create(**kwargs)
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    yield ChatLLMResponse(thinking=delta.reasoning)
                elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    yield ChatLLMResponse(thinking=delta.reasoning_content)
        except Exception as e:
            raise RuntimeError(f"OpenRouter API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of OpenRouter chat completion"""
        try:
            or_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": True,
                "extra_body": {
                    "reasoning":{
                            "effort": self.reasoning_effort
                        }
                }
            }
            
            if or_tools:
                kwargs["tools"] = or_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = await self.async_client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    yield ChatLLMResponse(thinking=delta.reasoning)
                elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    yield ChatLLMResponse(thinking=delta.reasoning_content)
        except Exception as e:
            raise RuntimeError(f"OpenRouter API streaming error: {str(e)}")

    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata from OpenRouter"""
        try:
            response = requests.get(
                f"{self.base_url}/models/",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            data = response.json()
            models = data.get('data', [])
            target_model = [m for m in models if m.get("id") == self.model]
            
            if target_model:
                model_info = target_model[0]
                return ModelMetadata(
                    name=self.model,
                    context_window=model_info.get("context_length", 128000),
                    owned_by=self.model.split("/")[0] if "/" in self.model else "openrouter"
                )
        except Exception:
            pass
            
        # Fallback
        return ModelMetadata(
            name=self.model,
            context_window=128000,
            owned_by=self.model.split("/")[0] if "/" in self.model else "openrouter"
        )
