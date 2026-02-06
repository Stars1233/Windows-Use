import asyncio
import os
import logging
from typing import List, Any
from ..channels.base import BaseChannel
from ..agent.base import BaseAgent

logger = logging.getLogger("Gateway")

class Gateway:
    """
    Generalized Gateway that accepts signals from multiple channels
    and processes them sequentially through an Agent.
    """
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.task_queue = asyncio.Queue()
        self.channels: List[BaseChannel] = []
        self._bridge_tasks = {} # Track bridge tasks per channel
        self._current_task = None
        self._stop_event = asyncio.Event()

    def add_channel(self, channel: BaseChannel):
        """Register a new communication channel (WhatsApp, Signal, etc.)"""
        self.channels.append(channel)
        logger.info(f"Channel registered: {channel.__class__.__name__}")

    def remove_channel(self, channel: BaseChannel):
        """Stops and removes a specific channel from the gateway."""
        if channel in self.channels:
            logger.info(f"Removing channel: {channel.__class__.__name__}")
            
            # 1. Stop the transport
            try:
                channel.stop()
            except Exception as e:
                logger.error(f"Error stopping channel: {e}")
            
            # 2. Cancel the bridging task
            if channel in self._bridge_tasks:
                self._bridge_tasks[channel].cancel()
                del self._bridge_tasks[channel]
                
            # 3. Remove from list
            self.channels.remove(channel)

    async def start(self):
        """Starts the gateway and all registered channels."""
        logger.info("Starting Windows-Use Gateway...")
        
        # 1. Start all channels in the background
        for channel in self.channels:
            # We assume each channel has a .listen_for_messages() generator
            task = asyncio.create_task(self._bridge_channel(channel))
            self._bridge_tasks[channel] = task
            
            # Auto-start the transport if it has a .start() method
            logger.info(f"Starting transport for {channel.__class__.__name__}...")
            asyncio.create_task(asyncio.to_thread(channel.start))

        # 2. Sequential Processing Loop
        # This loop ensures the Agent is only handling one task at a time.
        while not self._stop_event.is_set():
            # Wait for next task in the unified queue
            channel, chat_jid, task, media_path, original_msg = await self.task_queue.get()
            
            try:
                self._current_task = task
                logger.info(f"Processing signal from {channel.__class__.__name__}: {task}")
                
                # Execute the actual Agent logic
                # For now, we use the transport's system command helper, 
                # but in future, this calls the Core Agent.
                result = await self._execute_agent_logic(channel, task, media_path)
                
                # Send result back to the specific channel it came from
                channel.send_response(chat_jid, result, original_msg)
                
            except Exception as e:
                logger.error(f"Gateway Error: {e}")
                channel.send_response(chat_jid, f"Error: {str(e)}", original_msg)
            finally:
                self._current_task = None
                self.task_queue.task_done()
                
                # Cleanup media if any
                if media_path and os.path.exists(media_path):
                    try: os.remove(media_path)
                    except: pass

    async def _bridge_channel(self, channel: Any):
        """Continuously pulls messages from a channel and puts them in the central queue."""
        async for chat_jid, task, media_path, original_msg in channel.listen_for_messages():
            logger.info(f"Signal Captured: {channel.__class__.__name__} -> Gateway Queue")
            await self.task_queue.put((channel, chat_jid, task, media_path, original_msg))

    async def _execute_agent_logic(self, channel, task, media_path):
        """
        The Agent's brain. Calls the self.agent.invoke method.
        """
        try:
            # We run it in a thread since Agent.invoke is synchronous and can be heavy
            logger.info(f"Invoking Agent for task: {task}")
            
            # Note: We should handle media_path if Agent supports it, 
            # currently Agent.invoke only takes query.
            # We can prefix the query with image info if needed.
            if media_path:
                task = f"[Image Attached: {media_path}] {task}"
                
            result = await asyncio.to_thread(self.agent.invoke, task)
            
            if hasattr(result, "content") and result.content:
                return result.content
            if hasattr(result, "error") and result.error:
                return f"Agent Error: {result.error}"
            return "Task completed with no output."
        except Exception as e:
            return f"Agent Execution Failed: {str(e)}"

    def stop(self):
        logger.info("Stopping Gateway and channels...")
        for channel in self.channels:
            try:
                channel.stop()
            except Exception as e:
                logger.error(f"Error stopping {channel.__class__.__name__}: {e}")
        self._stop_event.set()
