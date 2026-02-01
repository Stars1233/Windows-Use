from mistralai import HttpClient, AsyncHttpClient, RetryConfig, OptionalNullable, UserMessage, AssistantMessage, SystemMessage as MainMessage, TextChunk, ThinkChunk, ImageURL, ImageURLChunk, ResponseFormat, JSONSchema, ToolMessage as MistralToolMessage
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from typing import Union, Dict, Type, Iterator, AsyncIterator, Literal
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from dataclasses import dataclass
from mistralai import Mistral
from pydantic import BaseModel
import logging
import json
import os

class ChatMistral(BaseChatLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        server: Union[str, None] = None,
        server_url: Union[str, None] = None,
        url_params: Dict[str, str] = None,
        client: Type[HttpClient] = None,
        async_client: Type[AsyncHttpClient] = None,
        retry_config: OptionalNullable[RetryConfig] = None,
        timeout_ms: Union[int, None] = None,
        debug_logger: Union[logging.Logger, None] = None,
        reasoning_effort: Literal["none", "low", "medium", "high"] = "medium",
        top_p: float | None = None
    ):
        self.model = model
        if not api_key and not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY is not set")
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.temperature = temperature
        self.server = server
        self.max_tokens = max_tokens
        self.server_url = server_url
        self.url_params = url_params
        self._client = client
        self._async_client = async_client
        self.retry_config = retry_config
        self.timeout_ms = timeout_ms
        self.debug_logger = debug_logger
        self.reasoning_effort = reasoning_effort
        self.top_p = top_p

    @property
    def client(self) -> Mistral:
        """Initialize Mistral client"""
        kwargs = {"api_key": self.api_key}
        
        if self.server:
            kwargs["server"] = self.server
        if self.server_url:
            kwargs["server_url"] = self.server_url
        if self.url_params:
            kwargs["url_params"] = self.url_params
        if self._client:
            kwargs["client"] = self._client
        if self._async_client:
            kwargs["async_client"] = self._async_client
        if self.retry_config:
            kwargs["retry_config"] = self.retry_config
        if self.timeout_ms:
            kwargs["timeout_ms"] = self.timeout_ms
        if self.debug_logger:
            kwargs["debug_logger"] = self.debug_logger
        
        return Mistral(**kwargs)

    @property
    def provider(self) -> str:
        return "mistral"
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def serialize_messages(self, messages: list[BaseMessage]) -> list:
        """Convert BaseMessage objects to Mistral message format"""
        serialized = []
        for message in messages:
            if isinstance(message, SystemMessage):
                content = [TextChunk(text=message.content)]
                serialized.append(MainMessage(content=content))
            elif isinstance(message, HumanMessage):
                content = [TextChunk(text=message.content)]
                serialized.append(UserMessage(content=content))
            elif isinstance(message, AIMessage):
                content = []
                if message.thinking:
                    content.append(ThinkChunk(thinking=[TextChunk(text=message.thinking)]))
                content.append(TextChunk(text=message.content))
                serialized.append(AssistantMessage(content=content))
            elif isinstance(message, ToolMessage):
                # Assistant Tool Call
                content = []
                if message.thinking:
                    content.append(ThinkChunk(thinking=[TextChunk(text=message.thinking)]))
                serialized.append(AssistantMessage(
                    content=content,
                    tool_calls=[{
                        "id": message.id,
                        "type": "function",
                        "function": {
                            "name": message.name,
                            "arguments": json.dumps(message.params)
                        }
                    }]
                ))
                # Tool Result (if content exists)
                if message.content:
                    serialized.append(MistralToolMessage(
                        tool_name=message.name,
                        tool_call_id=message.id,
                        content=str(message.content)
                    ))
            elif isinstance(message, ImageMessage):
                message.scale_images(scale=0.7)
                images = message.convert_images("base64")
                content = [TextChunk(text=message.content)]
                
                for image in images:
                    # Ensure proper data URL format
                    if not image.startswith('data:'):
                        image = f"data:{message.mime_type};base64,{image}"
                    
                    content.append(ImageURLChunk(
                        type="image_url",
                        image_url=ImageURL(url=image, detail="auto")
                    ))
                
                serialized.append(UserMessage(role="user", content=content))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return serialized
    
    def _prepare_tools(self, tools: list[Tool]) -> list[dict] | None:
        """Convert tools to Mistral format"""
        if not tools:
            return None
        
        return [{'type': 'function', 'function': tool.json_schema} for tool in tools]
    
    def _prepare_response_format(self, structured_output: BaseModel | None, json_mode: bool) -> ResponseFormat | None:
        """Prepare response format for structured output or JSON mode"""
        if structured_output:
            return ResponseFormat(
                json_schema=JSONSchema(
                    name=structured_output.__class__.__name__,
                    description="Model output structured as JSON schema",
                    schema_definition=structured_output.model_json_schema()
                ),
                type="json_schema"
            )
        elif json_mode:
            return ResponseFormat(type="json_object")
        return None
    
    def _extract_content_and_thinking(self, ai_contents) -> tuple[list[str], list[str]]:
        """Extract text content and thinking from response"""
        thinking_parts = []
        content_parts = []
        
        if isinstance(ai_contents, str):
            content_parts.append(ai_contents)
        elif isinstance(ai_contents, list):
            for ai_content in ai_contents:
                if isinstance(ai_content, TextChunk):
                    content_parts.append(ai_content.text)
                elif isinstance(ai_content, ThinkChunk):
                    # Handle thinking whether it's a string or a list of chunks
                    if isinstance(ai_content.thinking, str):
                        thinking_parts.append(ai_content.thinking)
                    elif isinstance(ai_content.thinking, list):
                        for chunk in ai_content.thinking:
                            if hasattr(chunk, 'text'):
                                thinking_parts.append(chunk.text)
                else:
                    raise ValueError(f"Unsupported content type: {type(ai_content)}")
        else:
            raise ValueError(f"Unsupported content type: {type(ai_contents)}")
        
        return content_parts, thinking_parts
    
    def _parse_response(self, completion, structured_output: BaseModel | None = None) -> tuple[BaseMessage | BaseModel, str | None]:
        """Parse Mistral response into content and thinking"""
        if structured_output:
            try:
                content = structured_output.model_validate_json(completion.choices[0].message.content)
                thinking = None
            except Exception as e:
                raise ValueError(f"Failed to parse structured output: {e}")
        else:
            ai_contents = completion.choices[0].message.content
            content_parts, thinking_parts = self._extract_content_and_thinking(ai_contents)
            
            if completion.choices[0].message.tool_calls:
                tool_call = completion.choices[0].message.tool_calls[0]
                content = ToolMessage(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    params=json.loads(tool_call.function.arguments),
                    content=None,
                    thinking="".join(thinking_parts) if thinking_parts else None
                )
            else:
                content = AIMessage(
                    content="".join(content_parts),
                    thinking="".join(thinking_parts) if thinking_parts else None
                )
            
            thinking = "".join(thinking_parts) if thinking_parts else None
        
        return content, thinking
    
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Synchronous invocation of Mistral chat completion"""
        try:
            mistral_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": False,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if mistral_tools:
                kwargs["tools"] = mistral_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            completion = self.client.chat.complete(**kwargs)
            
            content, thinking = self._parse_response(completion, structured_output)
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
            raise RuntimeError(f"Mistral API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of Mistral chat completion"""
        try:
            mistral_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
                "stream": False,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if mistral_tools:
                kwargs["tools"] = mistral_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            completion = await self.client.chat.complete_async(**kwargs)
            
            content, thinking = self._parse_response(completion, structured_output)
            
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
            raise RuntimeError(f"Mistral API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of Mistral chat completion"""
        try:
            mistral_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if mistral_tools:
                kwargs["tools"] = mistral_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = self.client.chat.stream(**kwargs)
            
            for chunk in stream:
                delta = chunk.data.choices[0].delta
                if delta.content:
                    if isinstance(delta.content, str):
                        yield ChatLLMResponse(content=AIMessage(content=delta.content))
                    elif isinstance(delta.content, list):
                        for part in delta.content:
                            if isinstance(part, TextChunk) and part.text:
                                yield ChatLLMResponse(content=AIMessage(content=part.text))
                            elif isinstance(part, ThinkChunk):
                                if isinstance(part.thinking, str) and part.thinking:
                                    yield ChatLLMResponse(thinking=part.thinking)
        except Exception as e:
            raise RuntimeError(f"Mistral API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of Mistral chat completion"""
        try:
            mistral_tools = self._prepare_tools(tools)
            response_format = self._prepare_response_format(structured_output, json_mode)
            
            kwargs = {
                "model": self.model,
                "messages": self.serialize_messages(messages),
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if mistral_tools:
                kwargs["tools"] = mistral_tools
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = await self.client.chat.stream_async(**kwargs)
            
            async for chunk in stream:
                delta = chunk.data.choices[0].delta
                if delta.content:
                    if isinstance(delta.content, str):
                        yield ChatLLMResponse(content=AIMessage(content=delta.content))
                    elif isinstance(delta.content, list):
                        for part in delta.content:
                            if isinstance(part, TextChunk) and part.text:
                                yield ChatLLMResponse(content=AIMessage(content=part.text))
                            elif isinstance(part, ThinkChunk):
                                if isinstance(part.thinking, str) and part.thinking:
                                    yield ChatLLMResponse(thinking=part.thinking)
        except Exception as e:
            raise RuntimeError(f"Mistral API streaming error: {str(e)}")

    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata from Mistral"""
        try:
            response = self.client.models.retrieve(model_id=self.model)
            return ModelMetadata(
                name=self.model,
                context_window=response.max_context_length,
                owned_by="mistral"
            )
        except Exception as e:
            # Fallback to default context windows for known models
            context_windows = {
                "mistral-large-latest": 128000,
                "mistral-large-2411": 128000,
                "mistral-small-latest": 32000,
                "mistral-small-2409": 32000,
                "codestral-latest": 32000,
                "ministral-8b-latest": 128000,
                "ministral-3b-latest": 128000,
                "pixtral-12b-latest": 128000,
            }
            
            context_window = 32000  # Default
            for key, window in context_windows.items():
                if key in self.model.lower():
                    context_window = window
                    break
            
            return ModelMetadata(
                name=self.model,
                context_window=context_window,
                owned_by="mistral"
            )