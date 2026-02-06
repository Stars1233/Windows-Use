import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Tuple, Any, Optional

class BaseChannel(ABC):
    """
    Abstract Base Class for all communication channels (WhatsApp, Signal, Telegram, etc.)
    """

    @abstractmethod
    async def listen_for_messages(self) -> AsyncGenerator[Tuple[str, str, Optional[str], Any], None]:
        """
        Async generator that yields signals to the Gateway.
        Yields: (chat_jid, task_text, media_path, original_message_object)
        """
        pass

    @abstractmethod
    def send_response(self, chat_jid: str, text: str, original_message: Optional[Any] = None) -> None:
        """
        Sends a response back to the user through the channel.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Initializes and connects the channel transport.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Disconnects and cleans up the channel transport.
        """
        pass
