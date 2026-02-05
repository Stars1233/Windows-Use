import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, Union, overload
from mistralai import Mistral
from mistralai.models import AssistantMessage, UserMessage, SystemMessage as MistralSystemMessage, ToolMessage as MistralToolMessage
from pydantic import BaseModel
from windows_use.llms.base import BaseChatLLM
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from windows_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from windows_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatMistral(BaseChatLLM):
    """
    Mistral AI LLM implementation following the BaseChatLLM protocol.
    
    Supports Mistral models including:
    - Mistral Large
    - Mistral Medium
    - Mistral Small
    - Codestral
    """
    
    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Mistral LLM.

        Args:
            model (str): The model name to use.
            api_key (str, optional): Mistral API key. Defaults to MISTRAL_API_KEY environment variable.
            base_url (str, optional): Base URL for the API.
            timeout (int, optional): Request timeout.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments for chat completions.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.temperature = temperature
        self.client = Mistral(api_key=self.api_key, server_url=base_url)
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "mistral"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Any]:
        """
        Convert BaseMessage objects to Mistral-compatible message objects.
        """
        mistral_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                mistral_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                mistral_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                
                b64_imgs = msg.convert_images(format="base64")
                for b64 in b64_imgs:
                    content.append({
                        "type": "image_url",
                        "image_url": f"data:{msg.mime_type};base64,{b64}"
                    })
                mistral_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                mistral_messages.append({"role": "assistant", "content": msg.content or ""})
            elif isinstance(msg, ToolMessage):
                # Mistral requires assistant message with tool_calls followed by tool message
                mistral_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": msg.id,
                        "type": "function",
                        "function": {
                            "name": msg.name,
                            "arguments": json.dumps(msg.params)
                        }
                    }]
                })
                mistral_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "name": msg.name,
                    "content": msg.content or ""
                })
        return mistral_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to Mistral-compatible tool definitions.
        """
        return [
            {
                "type": "function",
                "function": tool.json_schema
            }
            for tool in tools
        ]

    def _process_response(self, response: Any) -> ChatLLMResponse:
        """
        Process Mistral API response into ChatLLMResponse object.
        """
        choice = response.choices[0]
        message = choice.message
        usage_data = response.usage
        
        usage = ChatLLMUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens
        )
        
        content = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                # Mistral may return arguments as string or dict
                if isinstance(tool_call.function.arguments, str):
                    params = json.loads(tool_call.function.arguments)
                elif isinstance(tool_call.function.arguments, dict):
                    params = tool_call.function.arguments
                else:
                    params = {}
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse tool arguments: {e}")
                params = {}
                
            content = ToolMessage(
                id=tool_call.id,
                name=tool_call.function.name,
                params=params
            )
        else:
            content = AIMessage(content=message.content or "")
            
        return ChatLLMResponse(
            content=content,
            usage=usage
        )

    @overload
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        mistral_messages = self._convert_messages(messages)
        mistral_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": mistral_messages,
            **self.kwargs
        }
        
        # Only add tools if they exist
        if mistral_tools:
            params["tools"] = mistral_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output or json_mode:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.complete(**params)
        
        if structured_output:
            try:
                # Parse JSON response into structured output
                content_text = response.choices[0].message.content
                if content_text:
                    parsed = structured_output.model_validate_json(content_text)
                else:
                    parsed = structured_output()
                    
                return ChatLLMResponse(
                    content=parsed,
                    usage=ChatLLMUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens
                    )
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")
                # Fall through to normal processing

        return self._process_response(response)

    @overload
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        mistral_messages = self._convert_messages(messages)
        mistral_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": mistral_messages,
            **self.kwargs
        }
        
        if mistral_tools:
            params["tools"] = mistral_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output or json_mode:
            params["response_format"] = {"type": "json_object"}

        response = await self.client.chat.complete_async(**params)
        
        if structured_output:
            try:
                content_text = response.choices[0].message.content
                if content_text:
                    parsed = structured_output.model_validate_json(content_text)
                else:
                    parsed = structured_output()
                    
                return ChatLLMResponse(
                    content=parsed,
                    usage=ChatLLMUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens
                    )
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse structured output: {e}")
        
        return self._process_response(response)

    @overload
    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ...

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        mistral_messages = self._convert_messages(messages)
        mistral_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": mistral_messages,
            **self.kwargs
        }
        
        if mistral_tools:
            params["tools"] = mistral_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.stream(**params)
        
        for chunk in response:
            if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices') and chunk.data.choices:
                delta = chunk.data.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        mistral_messages = self._convert_messages(messages)
        mistral_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": mistral_messages,
            **self.kwargs
        }
        
        if mistral_tools:
            params["tools"] = mistral_tools
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.stream_async(**params)
        
        async for chunk in response:
            if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices') and chunk.data.choices:
                delta = chunk.data.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield ChatLLMResponse(content=AIMessage(content=delta.content))

    def get_metadata(self) -> Metadata:
        # Context windows vary by model
        context_window = 128000  # Default
        
        if "large" in self._model:
            context_window = 128000
        elif "medium" in self._model:
            context_window = 32000
        elif "small" in self._model:
            context_window = 32000
        elif "codestral" in self._model:
            context_window = 32000
        
        return Metadata(
            name=self._model,
            context_window=context_window,
            owned_by="mistral"
        )