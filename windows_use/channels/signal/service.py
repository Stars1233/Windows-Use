import os
import asyncio
import logging
import json
import threading
from typing import AsyncGenerator, Tuple, Any, Optional
from pathlib import Path
import httpx
import websockets
from ..base import BaseChannel

logger = logging.getLogger("Signal")

class SignalChannel(BaseChannel):
    """
    Signal Channel implementation using signal-cli-rest-api.
    Expected to work with https://github.com/bbernhard/signal-cli-rest-api
    """
    def __init__(self, 
                 rest_api_url: str = "http://localhost:8080",
                 phone_number: str = None,
                 bot_name: str = "🤖 *Windows Use*"):
        self.rest_api_url = rest_api_url.rstrip("/")
        self.bot_name = bot_name
        self.phone_number = phone_number
        self._msg_queue = asyncio.Queue()
        self._loop = None
        self._stop_event = asyncio.Event()
        self._auth_lock = threading.Lock()
        
        # Setup Session Directory
        user_home = Path(os.path.expanduser("~"))
        self.session_dir = user_home / ".windows-use" / "channels" / "signal"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.session_dir / "config.json"
        
        self.authorized_user_id = None
        self._load_config()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.phone_number = self.phone_number or config.get("phone_number")
                    self.authorized_user_id = config.get("authorized_user_id")
                    logger.info("Loaded Signal config.")
            except Exception as e:
                logger.error(f"Failed to load Signal config: {e}")

    def _save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "phone_number": self.phone_number,
                    "authorized_user_id": self.authorized_user_id
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save Signal config: {e}")

    async def _ensure_setup(self):
        if not self.phone_number:
            print("\n" + "╔" + "═"*48 + "╗")
            print("║" + " "*14 + "WINDOWS USE: SIGNAL SETUP" + " "*13 + "║")
            print("╠" + "═"*48 + "╣")
            print("║ 1. Ensure signal-cli-rest-api is running.      ║")
            print("║ 2. Enter your registered phone number.         ║")
            print("╚" + "═"*48 + "╝")
            self.phone_number = await asyncio.to_thread(input, "\n ➤ Signal Phone Number (e.g. +123456789): ")
            self.phone_number = self.phone_number.strip()
            if not self.phone_number:
                logger.error("No phone number provided. Signal channel will not function.")
                return False
            self._save_config()
        return True

    async def listen_for_messages(self) -> AsyncGenerator[Tuple[str, str, Optional[str], Any], None]:
        while True:
            chat_id, text, media_path, original_msg = await self._msg_queue.get()
            yield chat_id, text, media_path, original_msg

    def send_response(self, chat_jid: str, text: str, original_message: Optional[Any] = None) -> None:
        if not self._loop:
             logger.error("Signal loop not running.")
             return
             
        # Check if chat_jid is a group
        is_group = chat_jid.startswith("group.")
        
        formatted_text = f"{self.bot_name}\n\n{text}"
        
        asyncio.run_coroutine_threadsafe(
            self._async_send_response(chat_jid, formatted_text, is_group),
            self._loop
        )

    async def _async_send_response(self, chat_jid: str, text: str, is_group: bool):
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "message": text,
                    "number": self.phone_number,
                    "recipients": [chat_jid]
                }
                # Use v2/send for better compatibility
                endpoint = f"{self.rest_api_url}/v2/send"
                resp = await client.post(endpoint, json=data, timeout=30.0)
                if resp.status_code != 201 and resp.status_code != 200:
                    logger.error(f"Signal Send Failed: {resp.text}")
        except Exception as e:
            logger.error(f"Signal Send Error: {e}")

    async def _receive_loop(self):
        ws_url = self.rest_api_url.replace("http", "ws") + f"/v1/receive/{self.phone_number}"
        
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info(f"Connected to Signal WebSocket: {ws_url}")
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                            
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        if "envelope" not in data:
                            continue
                            
                        envelope = data["envelope"]
                        if "dataMessage" not in envelope:
                            continue
                            
                        msg = envelope["dataMessage"]
                        sender = envelope["source"]
                        # If source is missing, it might be a group message with sourceNumber or similar
                        if not sender and "sourceNumber" in envelope:
                            sender = envelope["sourceNumber"]
                        
                        text = msg.get("message", "")
                        group_info = msg.get("groupInfo", {})
                        chat_id = sender
                        
                        if group_info and "groupId" in group_info:
                            chat_id = group_info["groupId"]

                        # 1. Check Authorization
                        if self.authorized_user_id is None:
                            with self._auth_lock:
                                if self.authorized_user_id is None:
                                    print("\n" + "╔" + "═"*48 + "╗")
                                    print("║" + " "*14 + "🔐  AUTHORIZATION REQUEST" + " "*13 + "║")
                                    print("╠" + "═"*48 + "╣")
                                    print(f"║ User: {sender:<41}║")
                                    print("╚" + "═"*48 + "╝")
                                    
                                    choice = await asyncio.to_thread(input, "\n ➤ Authorize this user to control your PC? (y/n): ")
                                    
                                    if choice.lower() == 'y':
                                        self.authorized_user_id = sender
                                        self._save_config()
                                        await self._async_send_response(chat_id, "✅ *Authorization Successful!*\n\nI am now your Windows Agent.", False)
                                        print(f"User {sender} has been authorized.")
                                    else:
                                        await self._async_send_response(chat_id, "❌ You are not authorized.", False)
                                        continue

                        # 2. Verify identity for subsequent messages
                        if sender != self.authorized_user_id:
                            continue

                        # 3. Handle Media Downloads
                        media_path = None
                        attachments = msg.get("attachments", [])
                        if attachments:
                            try:
                                att = attachments[0]
                                att_id = att.get("id")
                                if att_id:
                                    # Download attachment via REST API
                                    async with httpx.AsyncClient() as client:
                                        # Note: different versions of signal-cli-rest-api might have different attachment endpoints
                                        # Standard is /v1/attachments/<id>
                                        att_resp = await client.get(f"{self.rest_api_url}/v1/attachments/{att_id}")
                                        if att_resp.status_code == 200:
                                            import tempfile
                                            ext = ".bin"
                                            content_type = att.get("contentType", "")
                                            if "image" in content_type: ext = ".jpg"
                                            elif "audio" in content_type: ext = ".ogg"
                                            
                                            fd, path = tempfile.mkstemp(suffix=ext)
                                            with os.fdopen(fd, 'wb') as tmp:
                                                tmp.write(att_resp.content)
                                            media_path = path
                                            logger.info(f"Downloaded Signal attachment to {media_path}")
                            except Exception as e:
                                logger.error(f"Failed to download Signal attachment: {e}")

                        # 4. Queue for Gateway
                        if text or media_path:
                            await self._msg_queue.put((chat_id, text, media_path, data))
                            
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Signal WebSocket error: {e}. Retrying in 10s...")
                    await asyncio.sleep(10)

    def start(self) -> None:
        logger.info("Starting Signal Channel...")
        self._loop = asyncio.get_event_loop()
        
        # We need to run the setup and receive loop in the background
        asyncio.create_task(self._main_task())

    async def _main_task(self):
        if await self._ensure_setup():
            # Check if API is reachable
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.rest_api_url}/v1/about")
                    if resp.status_code == 200:
                        logger.info(f"Signal REST API at {self.rest_api_url} is online.")
                    else:
                        logger.warning(f"Signal REST API returned {resp.status_code}")
            except Exception as e:
                logger.error(f"Could not connect to Signal REST API at {self.rest_api_url}: {e}")
                print(f"⚠️  Signal REST API is unreachable. Please ensure it is running at {self.rest_api_url}")
                return

            await self._receive_loop()

    def stop(self) -> None:
        logger.info("Stopping Signal Channel...")
        self._stop_event.set()
