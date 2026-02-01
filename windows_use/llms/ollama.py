from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from ollama import Client, AsyncClient, Image, Message
from typing import Iterator, AsyncIterator
from windows_use.llms.base import BaseChatLLM
from windows_use.tool.service import Tool
from dataclasses import dataclass
from pydantic import BaseModel
import json
import os

class ChatOllama(BaseChatLLM):
    def __init__(self, host: str | None = None, model: str | None = None, think: bool = False, temperature: float = 0.7, timeout: int | None = None):
        self.host = host or "http://localhost:11434"  # Default to local Ollama
        self.model = model
        self.think = think
        self.temperature = temperature
        self.timeout = timeout
    
    @property
    def provider(self) -> str:
        return "ollama"
    
    @property
    def client(self) -> Client:
        """Initialize synchronous Ollama client with optional timeout"""
        kwargs = {"host": self.host}
        if self.timeout:
            kwargs["timeout"] = self.timeout
        return Client(**kwargs)

    @property
    def async_client(self) -> AsyncClient:
        """Initialize asynchronous Ollama client with optional timeout"""
        kwargs = {"host": self.host}
        if self.timeout:
            kwargs["timeout"] = self.timeout
        return AsyncClient(**kwargs)
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def sanitize_schema(self, tool_schema: dict) -> dict:
        """
        Convert full JSON schema into Ollama-compatible minimal function schema.
        Keeps only: name, description, and simplified parameters.
        """
        params = tool_schema.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        clean_props = {}

        for name, prop in properties.items():
            if isinstance(prop, dict):
                t = prop.get("type", "string")
            else:
                t = "string"

            # Normalize to valid JSON schema types
            if t not in {"string", "integer", "number", "boolean", "array", "object"}:
                t = "string"

            clean_props[name] = {"type": t}

        return {
            "name": tool_schema.get("name"),
            "description": tool_schema.get("description"),
            "parameters": {
                "type": "object",
                "properties": clean_props,
                "required": required
            }
        }
    
    def serialize_messages(self, messages: list[BaseMessage]) -> list[Message]:
        """Convert BaseMessage objects to Ollama Message format"""
        serialized = []
        for message in messages:
            if isinstance(message, SystemMessage):
                serialized.append(Message(role="system", content=message.content))
            elif isinstance(message, HumanMessage):
                serialized.append(Message(role="user", content=message.content))
            elif isinstance(message, AIMessage):
                content = message.content
                if message.thinking:
                    content = f"<thought>\n{message.thinking}\n</thought>\n{content}"
                serialized.append(Message(role="assistant", content=content))
            elif isinstance(message, ToolMessage):
                # Assistant Tool Call
                content = None
                if message.thinking:
                    content = f"<thought>\n{message.thinking}\n</thought>"
                serialized.append(Message(
                    role="assistant",
                    content=content,
                    tool_calls=[{
                        "function": {
                            "name": message.name,
                            "arguments": message.params
                        }
                    }]
                ))
                # Tool Result (if content exists)
                if message.content:
                    serialized.append(Message(
                        role="tool",
                        name=message.name,
                        content=str(message.content)
                    ))
            elif isinstance(message, ImageMessage):
                message.scale_images(scale=0.7)
                images = message.convert_images("bytes")
                serialized.append(Message(
                    role="user", 
                    content=message.content, 
                    images=[Image(value=image) for image in images]
                ))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return serialized
    
    def _prepare_tools(self, tools: list[Tool]) -> list[dict] | None:
        """Convert tools to Ollama format with sanitized schemas"""
        if not tools:
            return None
        
        return [{
            'type': 'function', 
            'function': self.sanitize_schema(tool.json_schema)
        } for tool in tools]
    
    def _extract_usage(self, completion) -> ChatLLMUsage:
        """Extract usage statistics from completion response"""
        try:
            # Handle both dict and object attribute access
            if isinstance(completion, dict):
                prompt_tokens = completion.get("prompt_eval_count", 0)
                completion_tokens = completion.get("eval_count", 0)
            else:
                prompt_tokens = getattr(completion, "prompt_eval_count", 0)
                completion_tokens = getattr(completion, "eval_count", 0)
            
            return ChatLLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        except Exception:
            return ChatLLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    
    def _parse_response(self, completion, structured_output: BaseModel | None = None) -> tuple[BaseMessage | BaseModel, str | None]:
        """Parse Ollama response into content and thinking"""
        thinking = getattr(completion.message, 'reasoning_content', None)
        
        if structured_output:
            content = structured_output.model_validate_json(completion.message.content)
            thinking = None
        elif completion.message.tool_calls:
            tool_call = completion.message.tool_calls[0]
            content = ToolMessage(
                id="tool-call-id",
                name=tool_call.function.name,
                params=tool_call.function.arguments,
                content=None,
                thinking=thinking
            )
        else:
            content = AIMessage(content=completion.message.content, thinking=thinking)
        
        return content, thinking
    
    def invoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Synchronous invocation of Ollama chat completion"""
        try:
            ollama_tools = self._prepare_tools(tools)
            
            kwargs = {
                "model": self.model,
                "stream": False,
                "messages": self.serialize_messages(messages),
            }
            
            if ollama_tools:
                kwargs["tools"] = ollama_tools
            
            if structured_output:
                kwargs["format"] = structured_output.model_json_schema()
            elif json_mode:
                kwargs["format"] = "json"
            
            if self.temperature is not None:
                kwargs["options"] = {"temperature": self.temperature}
            
            completion = self.client.chat(**kwargs)
            
            content, thinking = self._parse_response(completion, structured_output)
            
            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=self._extract_usage(completion)
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}: {str(e)}")
    
    async def ainvoke(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of Ollama chat completion"""
        try:
            ollama_tools = self._prepare_tools(tools)
            
            kwargs = {
                "model": self.model,
                "stream": False,
                "messages": self.serialize_messages(messages),
            }
            
            if ollama_tools:
                kwargs["tools"] = ollama_tools
            
            if structured_output:
                kwargs["format"] = structured_output.model_json_schema()
            elif json_mode:
                kwargs["format"] = "json"
            
            if self.temperature is not None:
                kwargs["options"] = {"temperature": self.temperature}
            
            completion = await self.async_client.chat(**kwargs)
            
            content, thinking = self._parse_response(completion, structured_output)
            
            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=self._extract_usage(completion)
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}: {str(e)}")

    def stream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of Ollama chat completion"""
        try:
            ollama_tools = self._prepare_tools(tools)
            
            kwargs = {
                "model": self.model,
                "stream": True,
                "messages": self.serialize_messages(messages),
            }
            
            if ollama_tools:
                kwargs["tools"] = ollama_tools
            
            if structured_output:
                kwargs["format"] = structured_output.model_json_schema()
            elif json_mode:
                kwargs["format"] = "json"
            
            if self.temperature is not None:
                kwargs["options"] = {"temperature": self.temperature}
            
            stream = self.client.chat(**kwargs)
            
            for chunk in stream:
                if 'message' in chunk:
                    if 'content' in chunk['message'] and chunk['message']['content']:
                        yield ChatLLMResponse(content=AIMessage(content=chunk['message']['content']))
                    if 'reasoning_content' in chunk['message'] and chunk['message']['reasoning_content']:
                        yield ChatLLMResponse(thinking=chunk['message']['reasoning_content'])
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}: {str(e)}")

    async def astream(self, messages: list[BaseMessage], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of Ollama chat completion"""
        try:
            ollama_tools = self._prepare_tools(tools)
            
            kwargs = {
                "model": self.model,
                "stream": True,
                "messages": self.serialize_messages(messages),
            }
            
            if ollama_tools:
                kwargs["tools"] = ollama_tools
            
            if structured_output:
                kwargs["format"] = structured_output.model_json_schema()
            elif json_mode:
                kwargs["format"] = "json"
            
            if self.temperature is not None:
                kwargs["options"] = {"temperature": self.temperature}
            
            stream = await self.async_client.chat(**kwargs)
            
            async for chunk in stream:
                if 'message' in chunk:
                    if 'content' in chunk['message'] and chunk['message']['content']:
                        yield ChatLLMResponse(content=AIMessage(content=chunk['message']['content']))
                    if 'reasoning_content' in chunk['message'] and chunk['message']['reasoning_content']:
                        yield ChatLLMResponse(thinking=chunk['message']['reasoning_content'])
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}: {str(e)}")
        
    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata from Ollama"""
        try:
            response = self.client.show(model=self.model)
            context_keys = [key for key in response.modelinfo.keys() if key.endswith("context_length")]
            context_window = response.modelinfo.get(context_keys[0]) if context_keys else 4096
            return ModelMetadata(name=self.model, context_window=context_window, owned_by="ollama")
        except Exception as e:
            raise ConnectionError(f"Failed to get model specification from Ollama at {self.host}: {str(e)}")