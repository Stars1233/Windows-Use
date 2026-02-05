import os
import json
import logging
from typing import Iterator, AsyncIterator, List, Optional, Any, Union, overload
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel
from windows_use.llms.base import BaseChatLLM
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, Metadata
from windows_use.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from windows_use.tool import Tool

logger = logging.getLogger(__name__)

class ChatAnthropic(BaseChatLLM):
    """
    Anthropic LLM implementation following the BaseChatLLM protocol.
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Anthropic LLM.

        Args:
            model (str): The model name to use.
            api_key (str, optional): Anthropic API key.
            base_url (str, optional): Base URL for the API.
            timeout (float): Request timeout in seconds.
            max_retries (int): Maximum number of retries.
            temperature (float, optional): Sampling temperature.
            **kwargs: Additional arguments for messages.create.
        """
        self._model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.temperature = temperature
        
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.aclient = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: List[BaseMessage]) -> tuple[Optional[str], List[dict]]:
        """
        Convert BaseMessage objects to Anthropic-compatible message dictionaries.
        Returns (system_prompt, messages).
        """
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ImageMessage):
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                
                b64_imgs = msg.convert_images(format="base64")
                for b64 in b64_imgs:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": msg.mime_type,
                            "data": b64
                        }
                    })
                anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                content = []
                if msg.thinking:
                    content.append({"type": "thinking", "thinking": msg.thinking, "signature": msg.thinking_signature})
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                anthropic_messages.append({"role": "assistant", "content": content if content else ""})
            elif isinstance(msg, ToolMessage):
                # Anthropic requires a tool_use block followed by a tool_result block
                # However, our message history might be complex. 
                # We need to make sure we find the corresponding tool_use in the assistant message.
                
                # Check if the previous message was an assistant message with tool calls
                if anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                    # If it's already an assistant message, we check if it has tool_use
                    last_content = anthropic_messages[-1]["content"]
                    if isinstance(last_content, str):
                        last_content = [{"type": "text", "text": last_content}]
                    
                    tool_use_block = {
                        "type": "tool_use",
                        "id": msg.id,
                        "name": msg.name,
                        "input": msg.params
                    }
                    if isinstance(last_content, list):
                        last_content.append(tool_use_block)
                    anthropic_messages[-1]["content"] = last_content
                else:
                    # If no assistant message, we inject one (though this shouldn't happen with correct history)
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": msg.id,
                            "name": msg.name,
                            "input": msg.params
                        }]
                    })
                
                # Now add the tool result
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.id,
                        "content": msg.content or ""
                    }]
                })
                
        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: List[Tool]) -> List[dict]:
        """
        Convert Tool objects to Anthropic-compatible tool definitions.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.json_schema["parameters"]
            }
            for tool in tools
        ]

    def _process_response(self, response: Any) -> ChatLLMResponse:
        """
        Process Anthropic API response into ChatLLMResponse object.
        """
        content_blocks = response.content
        usage_data = response.usage
        
        usage = ChatLLMUsage(
            prompt_tokens=usage_data.input_tokens,
            completion_tokens=usage_data.output_tokens,
            total_tokens=usage_data.input_tokens + usage_data.output_tokens
        )
        
        text_content = ""
        thinking = None
        tool_message = None
        
        for block in content_blocks:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking = block.thinking
            elif block.type == "tool_use":
                # Take the first tool use as per Agent logic
                if not tool_message:
                    tool_message = ToolMessage(
                        id=block.id,
                        name=block.name,
                        params=block.input
                    )
        
        final_content = tool_message if tool_message else AIMessage(content=text_content)
        if thinking:
            final_content.thinking = thinking
        
        return ChatLLMResponse(
            content=final_content,
            thinking=thinking,
            usage=usage
        )

    @overload
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "system": system_prompt,
            "tools": anthropic_tools,
            "max_tokens": 4096, # Default max tokens
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            # Simulate structured output via tool calling
            params["tools"] = [{
                "name": "record_result",
                "description": f"Record the structured result: {structured_output.__name__}",
                "input_schema": structured_output.model_json_schema()
            }]
            params["tool_choice"] = {"type": "tool", "name": "record_result"}
            
            response = self.client.messages.create(**params)
            tool_use = next(b for b in response.content if b.type == "tool_use")
            parsed = structured_output(**tool_use.input)
            
            return ChatLLMResponse(
                content=parsed,
                usage=ChatLLMUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens
                )
            )

        response = self.client.messages.create(**params)
        return self._process_response(response)

    @overload
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        ...

    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "system": system_prompt,
            "tools": anthropic_tools,
            "max_tokens": 4096,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        if structured_output:
            params["tools"] = [{
                "name": "record_result",
                "description": f"Record the structured result: {structured_output.__name__}",
                "input_schema": structured_output.model_json_schema()
            }]
            params["tool_choice"] = {"type": "tool", "name": "record_result"}
            
            response = await self.aclient.messages.create(**params)
            tool_use = next(b for b in response.content if b.type == "tool_use")
            parsed = structured_output(**tool_use.input)
            
            return ChatLLMResponse(
                content=parsed,
                usage=ChatLLMUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens
                )
            )

        response = await self.aclient.messages.create(**params)
        return self._process_response(response)

    @overload
    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        ...

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "system": system_prompt,
            "tools": anthropic_tools,
            "max_tokens": 4096,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        with self.client.messages.stream(**params) as stream:
            for event in stream:
                if event.type == "text":
                    yield ChatLLMResponse(content=AIMessage(content=event.text))
                elif event.type == "input_json":
                     # Support partial tool call yield? 
                     # For now, we skip streaming for tool calls in Agent logic
                     pass

    @overload
    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        ...

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        
        params = {
            "model": self._model,
            "messages": anthropic_messages,
            "system": system_prompt,
            "tools": anthropic_tools,
            "max_tokens": 4096,
            **self.kwargs
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        
        async with self.aclient.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "text":
                    yield ChatLLMResponse(content=AIMessage(content=event.text))

    def get_metadata(self) -> Metadata:
        return Metadata(
            name=self._model,
            context_window=200000,
            owned_by="anthropic"
        )
