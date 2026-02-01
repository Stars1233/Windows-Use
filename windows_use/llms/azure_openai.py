from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam, ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, ChatCompletionSystemMessageParam
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from typing import Iterator, AsyncIterator, Literal
from dataclasses import dataclass
from pydantic import BaseModel
from httpx import Client
import json
import os

class ChatAzureOpenAI(BaseChatLLM):
    def __init__(
        self,
        endpoint: str,
        deployment_name: str,
        api_key: str | None = None,
        model: str | None = None,
        api_version: str = "2024-10-21",
        temperature: float = 0.7,
        max_retries: int = 3,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: Client | None = None,
        strict_response_validation: bool = False
    ):
        self.endpoint = endpoint.rstrip('/')
        self.deployment_name = deployment_name
        if not api_key and not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.model = model
        self.api_version = api_version
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.strict_response_validation = strict_response_validation

    @property
    def client(self) -> AzureOpenAI:
        """Initialize synchronous Azure OpenAI client"""
        kwargs = {
            "api_key": self.api_key,
            "azure_endpoint": self.endpoint,
            "api_version": self.api_version,
            "azure_deployment": self.deployment_name,
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
        
        return AzureOpenAI(**kwargs)

    @property
    def async_client(self) -> AsyncAzureOpenAI:
        """Initialize asynchronous Azure OpenAI client"""
        kwargs = {
            "api_key": self.api_key,
            "azure_endpoint": self.endpoint,
            "api_version": self.api_version,
            "azure_deployment": self.deployment_name,
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
        
        return AsyncAzureOpenAI(**kwargs)

    @property
    def provider(self) -> str:
        return "azure_openai"

    @property
    def model_name(self) -> str:
        return self.model or self.deployment_name

    def serialize_messages(self, messages: list[BaseMessage]) -> list:
        """Convert BaseMessage objects to Azure OpenAI message format"""
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
        """Convert tools to Azure OpenAI format"""
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

    def _parse_response(self, message, structured_output: BaseModel | None = None) -> tuple[BaseMessage | BaseModel, str | None]:
        """Parse Azure OpenAI response into content and thinking"""
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
            content = AIMessage(content=message.content, thinking=thinking)
        
        return content, thinking

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Synchronous invocation of Azure OpenAI chat completion"""
        try:
            azure_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.deployment_name,
                "messages": self.serialize_messages(messages),
            }
            if any(x in (self.model or self.deployment_name).lower() for x in ["o1", "o3"]):
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
            
            if azure_tools:
                kwargs["tools"] = azure_tools
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
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of Azure OpenAI chat completion"""
        try:
            azure_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.deployment_name,
                "messages": self.serialize_messages(messages),
            }
            if any(x in (self.model or self.deployment_name).lower() for x in ["o1", "o3"]):
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
            
            if azure_tools:
                kwargs["tools"] = azure_tools
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
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of Azure OpenAI chat completion"""
        try:
            azure_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.deployment_name,
                "messages": self.serialize_messages(messages),
                "stream": True,
            }
            if any(x in (self.model or self.deployment_name).lower() for x in ["o1", "o3"]):
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
            
            if azure_tools:
                kwargs["tools"] = azure_tools
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
            raise RuntimeError(f"Azure OpenAI API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of Azure OpenAI chat completion"""
        try:
            azure_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.deployment_name,
                "messages": self.serialize_messages(messages),
                "stream": True,
            }
            if any(x in (self.model or self.deployment_name).lower() for x in ["o1", "o3"]):
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
            
            if azure_tools:
                kwargs["tools"] = azure_tools
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
            raise RuntimeError(f"Azure OpenAI API streaming error: {str(e)}")

    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata"""
        # Common Azure OpenAI deployment context windows
        context_windows = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-35-turbo": 16385,
            "gpt-3.5-turbo": 16385,
        }
        
        # Try to determine context window from model name
        model_name = self.model or self.deployment_name
        context_window = 128000  # Default for most modern deployments
        
        for key, window in context_windows.items():
            if key in model_name.lower():
                context_window = window
                break
        
        return ModelMetadata(
            name=model_name,
            context_window=context_window,
            owned_by="microsoft"
        )