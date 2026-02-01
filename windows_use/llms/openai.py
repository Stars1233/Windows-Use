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
from openai import OpenAI, AsyncOpenAI
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from dataclasses import dataclass
from pydantic import BaseModel
from httpx import Client
import json
import os


class ChatOpenAI(BaseChatLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        websocket_base_url: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: int | None = None,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: Client | None = None,
        strict_response_validation: bool = False
    ):
        self.model = model
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.organization = organization
        self.project = project
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.websocket_base_url = websocket_base_url
        self.strict_response_validation = strict_response_validation

    @property
    def client(self) -> OpenAI:
        """Initialize synchronous OpenAI client"""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.websocket_base_url:
            kwargs["websocket_base_url"] = self.websocket_base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.organization:
            kwargs["organization"] = self.organization
        if self.project:
            kwargs["project"] = self.project
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return OpenAI(**kwargs)

    @property
    def async_client(self) -> AsyncOpenAI:
        """Initialize asynchronous OpenAI client"""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.organization:
            kwargs["organization"] = self.organization
        if self.project:
            kwargs["project"] = self.project
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return AsyncOpenAI(**kwargs)

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def serialize_messages(self, messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
        """Convert BaseMessage objects to OpenAI message format"""
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
        """Convert tools to OpenAI format"""
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
        """Parse OpenAI response into content and thinking"""
        thinking = getattr(message, 'reasoning_content', None)
        
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
        """Synchronous invocation of OpenAI chat completion"""
        try:
            openai_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
            }
            if "o1" in self.model or "o3" in self.model:
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.frequency_penalty is not None:
                    kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.presence_penalty
            
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            if openai_tools:
                kwargs["tools"] = openai_tools
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
                    total_tokens=completion.usage.total_tokens,
                    thinking_tokens=getattr(completion.usage.completion_tokens_details, 'reasoning_tokens', 0) if hasattr(completion.usage, 'completion_tokens_details') else 0
                )
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of OpenAI chat completion"""
        try:
            openai_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
            }
            if "o1" in self.model or "o3" in self.model:
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.frequency_penalty is not None:
                    kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.presence_penalty
            
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            if openai_tools:
                kwargs["tools"] = openai_tools
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
                    total_tokens=completion.usage.total_tokens,
                    thinking_tokens=getattr(completion.usage.completion_tokens_details, 'reasoning_tokens', 0) if hasattr(completion.usage, 'completion_tokens_details') else 0
                )
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of OpenAI chat completion"""
        try:
            openai_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "stream": True,
            }
            if "o1" in self.model or "o3" in self.model:
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.frequency_penalty is not None:
                    kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.presence_penalty
            
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            if openai_tools:
                kwargs["tools"] = openai_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = self.client.chat.completions.create(**kwargs)
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    yield ChatLLMResponse(thinking=delta.reasoning_content)
        except Exception as e:
            raise RuntimeError(f"OpenAI API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of OpenAI chat completion"""
        try:
            openai_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "stream": True,
            }
            if "o1" in self.model or "o3" in self.model:
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.frequency_penalty is not None:
                    kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.presence_penalty
            
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            if openai_tools:
                kwargs["tools"] = openai_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = await self.async_client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    yield ChatLLMResponse(thinking=delta.reasoning_content)
        except Exception as e:
            raise RuntimeError(f"OpenAI API streaming error: {str(e)}")

    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata"""
        try:
            response = self.client.models.retrieve(model=self.model)
            context_window = getattr(response, 'context_window', 128000)
        except Exception:
            # Fallback for known OpenAI models
            context_windows = {
                "gpt-4o": 128000,
                "gpt-4o-mini": 128000,
                "gpt-4-turbo": 128000,
                "gpt-4": 8192,
                "o1-preview": 128000,
                "o1-mini": 128000,
                "o1": 200000,
            }
            
            context_window = 128000  # Default
            for key, window in context_windows.items():
                if key in self.model.lower():
                    context_window = window
                    break
                    
        return ModelMetadata(
            name=self.model,
            context_window=context_window,
            owned_by="openai"
        )
