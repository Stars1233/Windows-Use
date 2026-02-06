import os
import asyncio
import logging
import json
import threading
from typing import AsyncGenerator, Tuple, Any, Optional
from pathlib import Path
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from ..base import BaseChannel

logger = logging.getLogger("Slack")

class SlackChannel(BaseChannel):
    def __init__(self, bot_token: str = None, app_token: str = None, bot_name: str = "🤖 *Windows Use*"):
        self.bot_token = bot_token
        self.app_token = app_token
        self.bot_name = bot_name
        self._msg_queue = asyncio.Queue()
        self._loop = None
        self._handler = None
        
        # Setup Session Directory
        user_home = Path(os.path.expanduser("~"))
        self.session_dir = user_home / ".windows-use" / "channels" / "slack"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.session_dir / "config.json"
        
        self.authorized_user_id = None
        self._load_config()
        
        # Slack App Setup
        self.app = AsyncApp(token=self.bot_token)
        self._setup_handlers()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.bot_token = self.bot_token or config.get("bot_token")
                    self.app_token = self.app_token or config.get("app_token")
                    self.authorized_user_id = config.get("authorized_user_id")
                    logger.info("Loaded Slack config.")
            except Exception as e:
                logger.error(f"Failed to load Slack config: {e}")

    def _save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "bot_token": self.bot_token,
                    "app_token": self.app_token,
                    "authorized_user_id": self.authorized_user_id
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save Slack config: {e}")

    async def _ensure_tokens(self):
        modified = False
        if not self.bot_token:
            self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        if not self.app_token:
            self.app_token = os.getenv("SLACK_APP_TOKEN")
            
        if not self.bot_token or not self.app_token:
            print("\n" + "╔" + "═"*48 + "╗")
            print("║" + " "*15 + "WINDOWS USE: SLACK SETUP" + " "*14 + "║")
            print("╠" + "═"*48 + "╣")
            print("║ 1. Create a Slack App at api.slack.com/apps.   ║")
            print("║ 2. Enable 'Socket Mode'.                       ║")
            print("║ 3. Add 'app_mentions:read', 'im:read',         ║")
            print("║    'chat:write', 'files:read' scopes.          ║")
            print("║ 4. Install App to Workspace to get Bot Token.  ║")
            print("║ 5. Get App Token from 'Basic Information'.     ║")
            print("╚" + "═"*48 + "╝")
            
            if not self.bot_token:
                self.bot_token = (await asyncio.to_thread(input, "\n ➤ Enter Slack Bot Token (xoxb-...): ")).strip()
                modified = True
            if not self.app_token:
                self.app_token = (await asyncio.to_thread(input, " ➤ Enter Slack App Token (xapp-...): ")).strip()
                modified = True
                
            if not self.bot_token or not self.app_token:
                logger.error("Tokens missing. Slack channel will not function.")
                return False
            
            if modified:
                self._save_config()
                # Re-init app with new token
                self.app = AsyncApp(token=self.bot_token)
                self._setup_handlers()
        return True

    def _setup_handlers(self):
        @self.app.event("app_mention")
        async def handle_mention(event, say):
            await self._handle_incoming(event, say)

        @self.app.event("message")
        async def handle_message(event, say):
            # Only handle DM messages
            if event.get("channel_type") == "im":
                await self._handle_incoming(event, say)

    async def _handle_incoming(self, event, say):
        user_id = event.get("user")
        if not user_id:
            return

        # 1. Check Authorization
        if self.authorized_user_id is None:
            print("\n" + "╔" + "═"*48 + "╗")
            print("║" + " "*14 + "🔐  AUTHORIZATION REQUEST" + " "*13 + "║")
            print("╠" + "═"*48 + "╣")
            print(f"║ User ID: {user_id:<38}║")
            print("╚" + "═"*48 + "╝")
            
            choice = await asyncio.to_thread(input, "\n ➤ Authorize this user to control your PC? (y/n): ")
            
            if choice.lower() == 'y':
                self.authorized_user_id = user_id
                self._save_config()
                await say(f"✅ *Authorization Successful!*\n\nI am now your Windows Agent.")
            else:
                await say("❌ You are not authorized.")
                return

        # 2. Verify identity
        if user_id != self.authorized_user_id:
            return

        # 3. Process Content
        text = event.get("text", "")
        # Remove mention markup if present
        import re
        text = re.sub(r'<@U[A-Z0-9]+>', '', text).strip()
        
        media_path = None
        files = event.get("files", [])
        if files:
            for file in files:
                if file.get("mimetype", "").startswith("image/"):
                    logger.info(f"Downloading Slack image: {file.get('name')}")
                    import tempfile
                    import httpx
                    
                    url = file.get("url_private_download")
                    suffix = "." + file.get("filetype", "png")
                    
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(url, headers={"Authorization": f"Bearer {self.bot_token}"})
                        if resp.status_code == 200:
                            fd, path = tempfile.mkstemp(suffix=suffix)
                            with os.fdopen(fd, 'wb') as tmp:
                                tmp.write(resp.content)
                            media_path = path
                            break # One image for now

        # 4. Queue for Gateway
        if text or media_path:
            await self._msg_queue.put((event.get("channel"), text, media_path, event))

    async def listen_for_messages(self) -> AsyncGenerator[Tuple[str, str, Optional[str], Any], None]:
        while True:
            chat_id, text, media_path, original_msg = await self._msg_queue.get()
            yield str(chat_id), text, media_path, original_msg

    def send_response(self, chat_jid: str, text: str, original_message: Optional[Any] = None) -> None:
        if not self._loop:
            return
            
        formatted_text = f"{self.bot_name}\n\n{text}"
        
        asyncio.run_coroutine_threadsafe(
            self._async_send_response(chat_jid, formatted_text, original_message),
            self._loop
        )

    async def _async_send_response(self, chat_jid: str, text: str, original_msg: Optional[Any] = None):
        try:
            # Use thread_ts to reply in thread if available
            thread_ts = None
            if original_msg:
                thread_ts = original_msg.get("ts")
                
            await self.app.client.chat_postMessage(
                channel=chat_jid,
                text=text,
                thread_ts=thread_ts
            )
        except Exception as e:
            logger.error(f"Slack Send Error: {e}")

    def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        asyncio.create_task(self._main_task())

    async def _main_task(self):
        if await self._ensure_tokens():
            logger.info("Starting Slack Socket Mode Handler...")
            self._handler = AsyncSocketModeHandler(self.app, self.app_token)
            await self._handler.start_async()

    def stop(self) -> None:
        logger.info("Stopping Slack Channel...")
        if self._handler:
            # SocketModeHandler doesn't have a very clean async stop in some versions, 
            # but we can try closing the underlying client.
            pass
