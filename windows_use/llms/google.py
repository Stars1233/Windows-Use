from google.genai.types import Part, Content, GenerateContentConfigDict, Modality, ToolDict, FunctionDeclarationDict, FunctionCall
from windows_use.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ImageMessage, ToolMessage
from windows_use.llms.views import ChatLLMResponse, ChatLLMUsage, ModelMetadata
from google.genai.client import Client, DebugConfig
from google.auth.credentials import Credentials
from typing import Iterator, AsyncIterator, Literal
from windows_use.llms.base import BaseChatLLM
from dataclasses import dataclass
from windows_use.tool.service import Tool
from google.genai import types
from pydantic import BaseModel
import asyncio
import json
import os


class EventLoopManager:
    """Manages event loop lifecycle to avoid 'Event loop is closed' errors"""
    _loop = None
    _thread_id = None

    @classmethod
    def get_loop(cls):
        """Get or create an event loop for the current thread"""
        import threading
        current_thread = threading.current_thread().ident
        
        # If we're on a different thread, we need a new loop
        if cls._thread_id != current_thread:
            cls._thread_id = current_thread
            try:
                cls._loop = asyncio.get_running_loop()
            except RuntimeError:
                cls._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(cls._loop)
        
        # If no loop exists for this thread, create one
        if cls._loop is None or cls._loop.is_closed():
            cls._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls._loop)
        
        return cls._loop


def run_async(coro):
    """
    Run an async coroutine safely without closing the event loop.
    This prevents 'Event loop is closed' errors on subsequent calls.
    """
    loop = EventLoopManager.get_loop()
    
    if loop.is_running():
        # We're already in an async context (e.g., Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
    
    # Run the coroutine without closing the loop
    return loop.run_until_complete(coro)


class ChatGoogle(BaseChatLLM):
    def __init__(
        self,
        model: str,
        thinking_budget: int = -1,
        api_key: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        credentials: Credentials | None = None,
        http_options: types.HttpOptions | types.HttpOptionsDict | None = None,
        debug_config: DebugConfig | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        top_p: float | None = None,
        top_k: int | None = None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    ):
        self.model = model
        if not api_key and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is not set")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.vertexai = vertexai
        self.temperature = temperature
        self.credentials = credentials
        self.project = project
        self.location = location
        self.http_options = http_options
        self.debug_config = debug_config
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        
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
                # Gemini thinking budget must be less than max_tokens
                self.thinking_budget = int(self.max_tokens * percentage)
                # Ensure it's at least some reasonable amount if enabled
                self.thinking_budget = max(1024, min(self.thinking_budget, self.max_tokens - 1024))
        else:
            self.thinking_budget = thinking_budget
            
        self._client = None
        self._async_client = None

    @property
    def provider(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return self.model

    def _create_client(self) -> Client:
        """Create a Google GenAI client"""
        kwargs = {"api_key": self.api_key}
        
        if self.vertexai is not None:
            kwargs["vertexai"] = self.vertexai
        if self.project:
            kwargs["project"] = self.project
        if self.location:
            kwargs["location"] = self.location
        if self.credentials:
            kwargs["credentials"] = self.credentials
        if self.http_options:
            kwargs["http_options"] = self.http_options
        if self.debug_config:
            kwargs["debug_config"] = self.debug_config
        
        return Client(**kwargs)

    @property
    def client(self) -> Client:
        """Get or create synchronous Google GenAI client"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def async_client(self) -> Client:
        """Get or create asynchronous Google GenAI client"""
        if self._async_client is None:
            self._async_client = self._create_client()
        return self._async_client

    def serialize_messages(self, messages: list[BaseMessage]) -> tuple[str | None, list[Content]]:
        """Convert BaseMessage objects to Google GenAI Content format"""
        serialized = []
        system_instruction = None
        
        for message in messages:
            if isinstance(message, SystemMessage):
                system_instruction = message.content
            elif isinstance(message, HumanMessage):
                serialized.append(Content(role="user", parts=[Part.from_text(text=message.content)]))
            elif isinstance(message, AIMessage):
                parts = []
                if message.thinking:
                    parts.append(Part(thought=message.thinking))
                parts.append(Part.from_text(text=message.content))
                serialized.append(Content(role="model", parts=parts))
            elif isinstance(message, ToolMessage):
                # Model's function call
                parts = []
                if message.thinking:
                    parts.append(Part(thought=message.thinking))
                parts.append(Part.from_function_call(name=message.name, args=message.params))
                serialized.append(Content(
                    role="model",
                    parts=parts
                ))
                # User's function response
                if message.content:
                    serialized.append(Content(
                        role="user",
                        parts=[Part.from_function_response(name=message.name, response={"response": message.content})]
                    ))
            elif isinstance(message, ImageMessage):
                message.scale_images(scale=0.7)
                images = message.convert_images("bytes")
                parts = [Part(text=message.content)]
                parts.extend([Part.from_bytes(data=image, mime_type=message.mime_type) for image in images])
                serialized.append(Content(role="user", parts=parts))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        
        return system_instruction, serialized

    def _prepare_tools(self, tools: list[Tool]) -> list[ToolDict] | None:
        """Convert tools to Google GenAI format"""
        if not tools:
            return None
        
        function_declarations = [
            FunctionDeclarationDict(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.json_schema['parameters']
            ) for tool in tools
        ]
        
        return [ToolDict(function_declarations=function_declarations)]

    def _prepare_config(
        self,
        system_instruction: str | None,
        tools: list[Tool],
        structured_output: BaseModel | None,
        json_mode: bool
    ) -> GenerateContentConfigDict:
        """Prepare generation config for Google GenAI"""
        config: GenerateContentConfigDict = {
            "temperature": self.temperature,
            "response_modalities": [Modality.TEXT],
        }

        if self.max_tokens:
            config["max_output_tokens"] = self.max_tokens
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.top_k is not None:
            config["top_k"] = self.top_k

        if self.thinking_budget > 0:
            config["thinking_config"] = {
                "include_thoughts": True,
                "thinking_budget": self.thinking_budget
            }
        
        if system_instruction:
            config["system_instruction"] = system_instruction
        
        # Set response format
        if structured_output or json_mode:
            config["response_mime_type"] = "application/json"
            if structured_output:
                config["response_json_schema"] = structured_output.model_json_schema(mode="serialization")
        else:
            config["response_mime_type"] = "text/plain"
        
        # Add tools
        google_tools = self._prepare_tools(tools)
        if google_tools:
            config["tools"] = google_tools
        
        return config

    def _extract_text_and_thinking(self, completion) -> tuple[str, str | None]:
        """Extract text content and thinking from completion"""
        text_parts = []
        thinking_parts = []
        
        if completion.candidates and completion.candidates[0].content and completion.candidates[0].content.parts:
            for part in completion.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                thought = getattr(part, 'thought', None)
                if thought:
                    thinking_parts.append(str(thought))
        
        text = "".join(text_parts)
        thinking = "".join(thinking_parts) if thinking_parts else None
        
        return text, thinking

    def _parse_response(self, completion, structured_output: BaseModel | None = None) -> tuple[BaseMessage | BaseModel, str | None]:
        """Parse Google GenAI response into content and thinking"""
        # Check for valid response
        if not completion.candidates or len(completion.candidates) == 0:
            error_msg = "Model returned empty response."
            if hasattr(completion, 'prompt_feedback'):
                error_msg += f" Prompt feedback: {completion.prompt_feedback}"
            raise ValueError(error_msg)
        
        candidate = completion.candidates[0]
        if not candidate.content or not candidate.content.parts:
            error_msg = f"Model candidate has no content. Finish reason: {candidate.finish_reason}"
            if hasattr(candidate, 'safety_ratings'):
                error_msg += f", Safety ratings: {candidate.safety_ratings}"
            raise ValueError(error_msg)
        
        # Parse based on response type
        if structured_output:
            try:
                content = structured_output.model_validate(completion.parsed)
                thinking = None
            except Exception as e:
                raise ValueError(f"Failed to parse structured output: {e}")
        elif getattr(completion, 'function_calls', None):
            text, thinking = self._extract_text_and_thinking(completion)
            function_call = completion.function_calls[0]
            content = ToolMessage(
                id="tool-call-id",
                name=function_call.name,
                params=function_call.args,
                content=None,
                thinking=thinking
            )
        else:
            text, thinking = self._extract_text_and_thinking(completion)
            content = AIMessage(content=text, thinking=thinking)
        
        return content, thinking

    def invoke(self, messages: list[BaseMessage] = [], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Synchronous invocation of Google GenAI chat completion"""
        try:
            system_instruction, contents = self.serialize_messages(messages)
            config = self._prepare_config(system_instruction, tools, structured_output, json_mode)
            
            completion = run_async(self.async_client.aio.models.generate_content(
                model=self.model,
                config=config,
                contents=contents
            ))
            
            content, thinking = self._parse_response(completion, structured_output)
            
            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=ChatLLMUsage(
                    prompt_tokens=completion.usage_metadata.prompt_token_count or 0,
                    completion_tokens=completion.usage_metadata.candidates_token_count or 0,
                    total_tokens=completion.usage_metadata.total_token_count or 0
                )
            )
        except Exception as e:
            raise RuntimeError(f"Google GenAI API error: {str(e)}")

    async def ainvoke(self, messages: list[BaseMessage] = [], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> ChatLLMResponse:
        """Asynchronous invocation of Google GenAI chat completion"""
        try:
            system_instruction, contents = self.serialize_messages(messages)
            config = self._prepare_config(system_instruction, tools, structured_output, json_mode)
            
            completion = await self.async_client.aio.models.generate_content(
                model=self.model,
                config=config,
                contents=contents
            )
            
            content, thinking = self._parse_response(completion, structured_output)
            
            return ChatLLMResponse(
                content=content,
                thinking=thinking,
                usage=ChatLLMUsage(
                    prompt_tokens=completion.usage_metadata.prompt_token_count or 0,
                    completion_tokens=completion.usage_metadata.candidates_token_count or 0,
                    total_tokens=completion.usage_metadata.total_token_count or 0
                )
            )
        except Exception as e:
            raise RuntimeError(f"Google GenAI API error: {str(e)}")

    def stream(self, messages: list[BaseMessage] = [], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> Iterator[ChatLLMResponse]:
        """Synchronous streaming of Google GenAI chat completion"""
        try:
            system_instruction, contents = self.serialize_messages(messages)
            config = self._prepare_config(system_instruction, tools, structured_output, json_mode)
            
            stream = self.client.models.generate_content_stream(
                model=self.model,
                config=config,
                contents=contents
            )
            
            for chunk in stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            yield ChatLLMResponse(content=AIMessage(content=part.text))
                        thought = getattr(part, 'thought', None)
                        if thought:
                            yield ChatLLMResponse(thinking=str(thought))
        except Exception as e:
            raise RuntimeError(f"Google GenAI API streaming error: {str(e)}")

    async def astream(self, messages: list[BaseMessage] = [], tools: list[Tool] = [], structured_output: BaseModel | None = None, json_mode: bool = False) -> AsyncIterator[ChatLLMResponse]:
        """Asynchronous streaming of Google GenAI chat completion"""
        try:
            system_instruction, contents = self.serialize_messages(messages)
            config = self._prepare_config(system_instruction, tools, structured_output, json_mode)
            
            stream = await self.async_client.aio.models.generate_content_stream(
                model=self.model,
                config=config,
                contents=contents
            )
            
            async for chunk in stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            yield ChatLLMResponse(content=AIMessage(content=part.text))
                        thought = getattr(part, 'thought', None)
                        if thought:
                            yield ChatLLMResponse(thinking=str(thought))
        except Exception as e:
            raise RuntimeError(f"Google GenAI API streaming error: {str(e)}")

    def get_model_specification(self) -> ModelMetadata:
        """Retrieve model metadata from Google GenAI"""
        try:
            response = run_async(self.async_client.aio.models.get(model=self.model))
            
            return ModelMetadata(
                name=self.model,
                context_window=response.input_token_limit,
                owned_by="google"
            )
        except Exception as e:
            # Fallback to default context windows for known models
            context_windows = {
                "gemini-2.0-flash-exp": 1000000,
                "gemini-2.5-flash-lite": 1000000,
                "gemini-1.5-pro": 2000000,
                "gemini-1.5-flash": 1000000,
                "gemini-1.0-pro": 32760,
            }
            
            context_window = 32760  # Default
            for key, window in context_windows.items():
                if key in self.model.lower():
                    context_window = window
                    break
            
            return ModelMetadata(
                name=self.model,
                context_window=context_window,
                owned_by="google"
            )