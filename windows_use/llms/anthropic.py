from anthropic.types import Message, MessageParam, ImageBlockParam, Base64ImageSourceParam, TextBlockParam, CacheControlEphemeralParam, ToolUseBlockParam, ToolResultBlockParam, ThinkingBlockParam
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from anthropic import Anthropic, AsyncAnthropic
from typing import Iterator, AsyncIterator
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Literal
from httpx import Client
import json
import os

class ChatAnthropic(BaseChatLLM):
    def __init__(self, model: str, api_key: str | None = None, thinking_budget: int = -1, reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium", temperature: float = 0.7, max_tokens: int = 8192, auth_token: str | None = None, base_url: str | None = None, timeout: float | None = None, max_retries: int = 3, default_headers: dict[str, str] | None = None, default_query: dict[str, object] | None = None, http_client: Client | None = None, strict_response_validation: bool = False):
        self.model = model
        if not api_key and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.auth_token = auth_token
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        
        # Calculate thinking budget based on reasoning effort if not explicitly set
        if thinking_budget == -1:
            if reasoning_effort == "none":
                self.thinking_budget = -1
            else:
                effort_map = {
                    "minimal": 0.10,
                    "low": 0.20,
                    "medium": 0.50,
                    "high": 0.80,
                    "xhigh": 0.95
                }
                percentage = effort_map.get(reasoning_effort, 0.50)
                # Anthropic thinking budget must be less than max_tokens
                # We subtract a small buffer (e.g., 20% or at least some tokens) for the actual response content
                # However, the user request says reasoning_effort allocates token portion.
                # We'll follow the percentage but cap it to avoid exhausting all tokens.
                self.thinking_budget = int(self.max_tokens * percentage)
                # Ensure it's at least 1024 if enabled, but not more than max_tokens - 1024
                self.thinking_budget = max(1024, min(self.thinking_budget, self.max_tokens - 1024))
        else:
            self.thinking_budget = thinking_budget
            
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.strict_response_validation = strict_response_validation

    @property
    def client(self):
        kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation
        }
        if self.auth_token:
            kwargs["auth_token"] = self.auth_token
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return Anthropic(**kwargs)
    
    @property
    def async_client(self):
        kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "_strict_response_validation": self.strict_response_validation
        }
        if self.auth_token:
            kwargs["auth_token"] = self.auth_token
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client
        
        return AsyncAnthropic(**kwargs)

    @property
    def provider(self):
        return "anthropic"

    @property
    def model_name(self):
        return self.model
    
    def serialize_messages(self, messages: list[BaseMessage]):
        system_instruction = None
        serialized = []
        for message in messages:
            if isinstance(message, SystemMessage):
                system_instruction = [TextBlockParam(
                    type="text", text=message.content,
                    cache_control=CacheControlEphemeralParam(type="ephemeral")
                )]
            elif isinstance(message, HumanMessage):
                content = [TextBlockParam(type="text", text=message.content)]
                serialized.append(MessageParam(role="user", content=content))
            elif isinstance(message, AIMessage):
                content = []
                if message.thinking:
                    content.append(ThinkingBlockParam(type="thinking", thinking=message.thinking, signature=message.thinking_signature or "dummy-signature"))
                content.append(TextBlockParam(type="text", text=message.content))
                serialized.append(MessageParam(role="assistant", content=content))
            elif isinstance(message, ToolMessage):
                # 1. The Assistant's Tool Use Request
                content = []
                if message.thinking:
                    content.append(ThinkingBlockParam(type="thinking", thinking=message.thinking, signature=message.thinking_signature or "dummy-signature"))
                content.append(ToolUseBlockParam(
                    type="tool_use",
                    id=message.id,
                    name=message.name,
                    input=message.params
                ))
                serialized.append(MessageParam(role="assistant", content=content))
                
                # 2. The User's Tool Result (if executed)
                if message.content is not None:
                    tool_result_block = ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=message.id,
                        content=str(message.content)
                    )
                    serialized.append(MessageParam(role="user", content=[tool_result_block]))

            elif isinstance(message, ImageMessage):
                message.scale_images(scale=0.7)
                images = message.convert_images("base64")
                content = [TextBlockParam(type="text", text=message.content)]
                
                for image in images:
                    # Remove data URL prefix if present
                    if image.startswith('data:'):
                        image = image.split(',', 1)[1]
                    
                    content.append(ImageBlockParam(
                        type="image",
                        source=Base64ImageSourceParam(
                            type="base64",
                            data=image,
                            media_type=message.mime_type
                        )
                    ))
                
                serialized.append(MessageParam(role="user", content=content))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return system_instruction, serialized
    
    def _prepare_tools(self, tools: list[Tool]) -> list[dict] | None:
        """Convert tools to Anthropic format"""
        if not tools:
            return None
        
        return [{
            "name": tool.json_schema["name"],
            "description": tool.json_schema["description"],
            "input_schema": tool.json_schema["parameters"]
        } for tool in tools]
    
    def _parse_response(self, completion: Message) -> tuple[str, ToolMessage | None, str | None, str | None]:
        """Parse Anthropic response into content, tool_message, thinking, and thinking_signature"""
        content = ""
        tool_message = None
        thinking = None
        thinking_signature = None
        
        for block in completion.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                thinking = block.thinking
                thinking_signature = getattr(block, 'signature', None)
            elif block.type == "tool_use":
                tool_message = ToolMessage(
                    id=block.id,
                    name=block.name,
                    params=block.input,
                    content=None,
                    thinking=thinking,
                    thinking_signature=thinking_signature
                )
        
        return content, tool_message, thinking, thinking_signature
    
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False):
        try:
            system_instruction, messages = self.serialize_messages(messages)
            anthropic_tools = self._prepare_tools(tools)

            kwargs = {
                "max_tokens": self.max_tokens,
                "model": self.model,
                "system": system_instruction,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            
            # Add thinking support for models that support it
            if self.thinking_budget > 0 and ("sonnet" in self.model.lower() or "3-7" in self.model.lower()):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                kwargs["temperature"] = 1.0
            
            completion = self.client.messages.create(**kwargs)

            if not isinstance(completion, Message):
                raise ValueError("Unexpected response type from Anthropic API")
            
            content, tool_message, thinking, thinking_signature = self._parse_response(completion)
            
            if tool_message:
                response_content = tool_message
            elif structured_output:
                # Parse structured output from JSON
                try:
                    response_content = structured_output.model_validate_json(content)
                except Exception as e:
                    raise ValueError(f"Failed to parse structured output: {e}")
            else:
                response_content = AIMessage(content=content, thinking=thinking, thinking_signature=thinking_signature)

            return ChatLLMResponse(
                content=response_content,
                thinking=thinking,
                usage=ChatLLMUsage(
                    prompt_tokens=completion.usage.input_tokens,
                    completion_tokens=completion.usage.output_tokens,
                    total_tokens=completion.usage.input_tokens + completion.usage.output_tokens
                )
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False):
        try:
            system_instruction, messages = self.serialize_messages(messages)
            anthropic_tools = self._prepare_tools(tools)

            kwargs = {
                "max_tokens": self.max_tokens,
                "model": self.model,
                "system": system_instruction,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            
            # Add thinking support for models that support it
            if self.thinking_budget > 0 and ("sonnet" in self.model.lower() or "3-7" in self.model.lower()):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                kwargs["temperature"] = 1.0

            completion = await self.async_client.messages.create(**kwargs)

            if not isinstance(completion, Message):
                raise ValueError("Unexpected response type from Anthropic API")
            
            content, tool_message, thinking, thinking_signature = self._parse_response(completion)
            
            if tool_message:
                response_content = tool_message
            elif structured_output:
                try:
                    response_content = structured_output.model_validate_json(content)
                except Exception as e:
                    raise ValueError(f"Failed to parse structured output: {e}")
            else:
                response_content = AIMessage(content=content, thinking=thinking, thinking_signature=thinking_signature)

            return ChatLLMResponse(
                content=response_content,
                thinking=thinking,
                usage=ChatLLMUsage(
                    prompt_tokens=completion.usage.input_tokens,
                    completion_tokens=completion.usage.output_tokens,
                    total_tokens=completion.usage.input_tokens + completion.usage.output_tokens
                )
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        try:
            system_instruction, messages = self.serialize_messages(messages)
            anthropic_tools = self._prepare_tools(tools)

            kwargs = {
                "max_tokens": self.max_tokens,
                "model": self.model,
                "system": system_instruction,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            
            if self.thinking_budget > 0 and ("sonnet" in self.model.lower() or "3-7" in self.model.lower()):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                kwargs["temperature"] = 1.0

            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield ChatLLMResponse(content=AIMessage(content=event.delta.text))
                        # Handle thinking deltas if available
                        elif hasattr(event.delta, 'thinking') and event.delta.thinking:
                            yield ChatLLMResponse(thinking=event.delta.thinking)
        except Exception as e:
            raise RuntimeError(f"Anthropic API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        try:
            system_instruction, messages = self.serialize_messages(messages)
            anthropic_tools = self._prepare_tools(tools)
            
            kwargs = {
                "max_tokens": self.max_tokens,
                "model": self.model,
                "system": system_instruction,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            
            if self.thinking_budget > 0 and ("sonnet" in self.model.lower() or "3-7" in self.model.lower()):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                kwargs["temperature"] = 1.0
                
            async with self.async_client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield ChatLLMResponse(content=AIMessage(content=event.delta.text))
                        elif hasattr(event.delta, 'thinking') and event.delta.thinking:
                            yield ChatLLMResponse(thinking=event.delta.thinking)
        except Exception as e:
            raise RuntimeError(f"Anthropic API streaming error: {str(e)}")

    def get_model_specification(self):
        # Model context windows as of latest known specs
        context_windows = {
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-haiku-20241022": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
        }
        
        context_window = context_windows.get(self.model, 200000)
        
        return ModelMetadata(
            name=self.model,
            context_window=context_window,
            owned_by="anthropic"
        )