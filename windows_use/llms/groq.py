from groq.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
from groq.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchemaJsonSchema, ResponseFormatResponseFormatJsonSchema
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from groq.types.chat.chat_completion_content_part_image_param import ImageURL
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from typing import Iterator, AsyncIterator, Literal
from windows_use.llms.base import BaseChatLLM
from dataclasses import dataclass
from pydantic import BaseModel
from groq import Groq, AsyncGroq
from windows_use.tool import Tool
from httpx import Client
import json
import os

class ChatGroq(BaseChatLLM):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int | None = None,
        reasoning_effort: Literal["none", "default", "low", "medium", "high", "minimal", "xhigh"] = "default",
        max_tokens: int | None = None,
        top_p: float | None = None,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: Client | None = None,
        strict_response_validation: bool = False
    ):
        self.model = model
        if not api_key and not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_url = base_url
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.strict_response_validation = strict_response_validation

    @property
    def client(self) -> Groq:
        """Initialize synchronous Groq client"""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation,
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
        
        return Groq(**kwargs)
    
    @property
    def async_client(self) -> AsyncGroq:
        """Initialize asynchronous Groq client"""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation,
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
        
        return AsyncGroq(**kwargs)

    @property
    def provider(self) -> str:
        return "groq"
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def serialize_messages(self, messages: list[BaseMessage]) -> list:
        """Convert BaseMessage objects to Groq message format"""
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
        """Convert tools to Groq format"""
        if not tools:
            return None
        
        return [{'type': 'function', 'function': tool.json_schema} for tool in tools]
    
    def _prepare_response_format(self, structured_output: BaseModel | None, json_mode: bool):
        """Prepare response format for structured output or JSON mode"""
        if structured_output:
            return ResponseFormatResponseFormatJsonSchema(
                json_schema=ResponseFormatResponseFormatJsonSchemaJsonSchema(
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
        """Parse Groq response into content and thinking"""
        thinking = getattr(message, 'reasoning_content', None)
        
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
        """Synchronous invocation of Groq chat completion"""
        try:
            groq_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
            }

            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            
            if self.reasoning_effort not in ["default", "none"] or "deepseek" in self.model.lower() or "qwen" in self.model.lower():
                kwargs["extra_body"] = {
                    "reasoning_effort": self.reasoning_effort
                }
            
            if groq_tools:
                kwargs["tools"] = groq_tools
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
            raise RuntimeError(f"Groq API error: {str(e)}")
    
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of Groq chat completion"""
        try:
            groq_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
            }

            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            
            if self.reasoning_effort not in ["default", "none"] or "deepseek" in self.model.lower() or "qwen" in self.model.lower():
                kwargs["extra_body"] = {
                    "reasoning_effort": self.reasoning_effort
                }
            
            if groq_tools:
                kwargs["tools"] = groq_tools
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
            raise RuntimeError(f"Groq API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of Groq chat completion"""
        try:
            groq_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": True,
            }

            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            
            if self.reasoning_effort not in ["default", "none"] or "deepseek" in self.model.lower() or "qwen" in self.model.lower():
                kwargs["extra_body"] = {
                    "reasoning_effort": self.reasoning_effort
                }
            
            if groq_tools:
                kwargs["tools"] = groq_tools
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
            raise RuntimeError(f"Groq API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of Groq chat completion"""
        try:
            groq_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": True,
            }

            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            
            if self.reasoning_effort not in ["default", "none"] or "deepseek" in self.model.lower() or "qwen" in self.model.lower():
                kwargs["extra_body"] = {
                    "reasoning_effort": self.reasoning_effort
                }
            
            if groq_tools:
                kwargs["tools"] = groq_tools
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
            raise RuntimeError(f"Groq API streaming error: {str(e)}")
    
    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata from Groq"""
        try:
            response = self.client.models.retrieve(model=self.model)
            return ModelMetadata(
                name=self.model,
                context_window=response.context_window,
                owned_by=response.owned_by
            )
        except Exception as e:
            # Fallback to default context windows for known models
            context_windows = {
                "llama-3.3-70b-versatile": 131072,
                "llama-3.1-70b-versatile": 131072,
                "llama-3.1-8b-instant": 131072,
                "llama3-70b-8192": 8192,
                "llama3-8b-8192": 8192,
                "mixtral-8x7b-32768": 32768,
                "gemma2-9b-it": 8192,
            }
            
            context_window = 8192  # Default
            for key, window in context_windows.items():
                if key in self.model.lower():
                    context_window = window
                    break
            
            return ModelMetadata(
                name=self.model,
                context_window=context_window,
                owned_by="groq"
            )