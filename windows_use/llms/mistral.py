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
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                params = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                params = tool_call.function.arguments if isinstance(tool_call.function.arguments, dict) else {}
                
            content = ToolMessage(
                id=tool_call.id,
                name=tool_call.function.name,
                params=params
            )
        else:
            content = AIMessage(content=message.content)
            
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
            "tools": mistral_tools,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.complete(**params)
        
        if structured_output:
            # Basic validation for structured output if Mistral supports it or via parsing
            import json
            parsed = structured_output.model_validate_json(response.choices[0].message.content)
            return ChatLLMResponse(
                content=parsed,
                usage=ChatLLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            )

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
            "tools": mistral_tools,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        response = await self.client.chat.complete_async(**params)
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
            "tools": mistral_tools,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        response = self.client.chat.stream(**params)
        for chunk in response:
            if chunk.data.choices[0].delta.content:
                yield ChatLLMResponse(content=AIMessage(content=chunk.data.choices[0].delta.content))

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        mistral_messages = self._convert_messages(messages)
        mistral_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": mistral_messages,
            "tools": mistral_tools,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        response = await self.client.chat.stream_async(**params)
        async for chunk in response:
            if chunk.data.choices[0].delta.content:
                yield ChatLLMResponse(content=AIMessage(content=chunk.data.choices[0].delta.content))

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._model,
            context_window=128000,
            owned_by="mistral"
        )
