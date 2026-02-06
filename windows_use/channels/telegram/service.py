import os
import asyncio
import logging
import json
import threading
from typing import AsyncGenerator, Tuple, Any, Optional
from pathlib import Path
from ..base import BaseChannel
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

logger = logging.getLogger("Telegram")

class TelegramChannel(BaseChannel):
    def __init__(self, session_dir: str = None, bot_name: str = "🤖 *Windows Use*"):
        self.bot_name = bot_name
        self._msg_queue = asyncio.Queue()
        self._loop = None
        self.application = None
        self._auth_lock = threading.Lock()
        
        # Setup Session Directory
        if session_dir:
            self.session_dir = Path(session_dir)
        else:
            user_home = Path(os.path.expanduser("~"))
            self.session_dir = user_home / ".windows-use" / "channels" / "telegram"
            
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.session_dir / "config.json"
        
        self.token = None
        self.authorized_user_id = None
        
        self._load_config()
        self._ensure_token()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.token = config.get("token")
                    self.authorized_user_id = config.get("authorized_user_id")
                    logger.info("Loaded Telegram config from file.")
            except Exception as e:
                logger.error(f"Failed to load Telegram config: {e}")

    def _save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "token": self.token,
                    "authorized_user_id": self.authorized_user_id
                }, f, indent=4)
            logger.info(f"Telegram config saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save Telegram config: {e}")

    def _ensure_token(self):
        if not self.token:
            # Check environment variable first
            self.token = os.getenv("TELEGRAM_BOT_TOKEN")
            
        if not self.token:
            print("\n" + "╔" + "═"*48 + "╗")
            print("║" + " "*14 + "WINDOWS USE: TELEGRAM SETUP" + " "*13 + "║")
            print("╠" + "═"*48 + "╣")
            print("║ 1. Message @BotFather on Telegram.             ║")
            print("║ 2. Create a new bot /newbot                    ║")
            print("║ 3. Copy the API Token provided.                ║")
            print("╚" + "═"*48 + "╝")
            self.token = input("\n ➤ Enter your Telegram Bot Token: ").strip()
            print(" " + "─"*48)
            if not self.token:
                logger.error("No token provided. Telegram channel will not function.")
                return
            self._save_config()

        try:
            self.application = Application.builder().token(self.token).build()
            self._setup_handlers()
        except Exception as e:
            logger.error(f"Failed to initialize Telegram Application: {e}")

    def _setup_handlers(self):
        if not self.application:
            return
        # Use a catch-all handler for authorization and processing
        self.application.add_handler(MessageHandler(filters.ALL, self._handle_all))

    async def _handle_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message:
            return

        # 1. Check Authorization
        if self.authorized_user_id is None:
            user = update.effective_user
            with self._auth_lock:
                # Double check inside lock
                if self.authorized_user_id is None:
                    print("\n" + "╔" + "═"*48 + "╗")
                    print("║" + " "*14 + "🔐  AUTHORIZATION REQUEST" + " "*13 + "║")
                    print("╠" + "═"*48 + "╣")
                    print(f"║ User: {user.first_name:<41}║")
                    print(f"║ Tag: @{user.username:<40}║")
                    print(f"║ ID: {user.id:<42}║")
                    print("╚" + "═"*48 + "╝")
                    
                    choice = await asyncio.to_thread(input, "\n ➤ Authorize this user to control your PC? (y/n): ")
                    
                    if choice.lower() == 'y':
                        self.authorized_user_id = user.id
                        self._save_config()
                        await update.message.reply_text(
                            f"✅ *Authorization Successful!*\n\nWelcome {user.first_name}. I am now your Windows Agent. Send me commands or images to get started.",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        print(f"User {user.id} has been authorized.")
                    else:
                        await update.message.reply_text("❌ You are not authorized to use this agent.")
                        return

        # 2. Verify identity for subsequent messages
        if update.effective_user.id != self.authorized_user_id:
            # Silently ignore other users after one is authorized
            return

        # 3. Process Content
        text = update.message.text or update.message.caption or ""
        media_path = None
        
        # Determine if it's a command
        is_audio = bool(update.message.voice or update.message.audio)
        if is_audio:
            text = "(audio)"

        # Handle Media Downloads
        try:
            if update.message.photo:
                logger.info("Downloading photo from Telegram...")
                photo_file = await update.message.photo[-1].get_file()
                import tempfile
                fd, path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                await photo_file.download_to_drive(path)
                media_path = path
            elif is_audio:
                logger.info("Downloading audio from Telegram...")
                msg = update.message.voice or update.message.audio
                voice_file = await msg.get_file()
                import tempfile
                fd, path = tempfile.mkstemp(suffix=".ogg")
                os.close(fd)
                await voice_file.download_to_drive(path)
                media_path = path
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
            await update.message.reply_text("⚠️ Failed to process media attachment.")

        # 4. Queue for Gateway
        if text or media_path:
            await self._msg_queue.put((update.effective_chat.id, text, media_path, update))

    async def listen_for_messages(self) -> AsyncGenerator[Tuple[str, str, Optional[str], Any], None]:
        while True:
            chat_id, text, media_path, original_msg = await self._msg_queue.get()
            yield str(chat_id), text, media_path, original_msg

    def send_response(self, chat_jid: str, text: str, original_message: Optional[Any] = None) -> None:
        if not self._loop or not self.application:
            logger.error("Telegram Channel not ready (loop or application missing).")
            return

        formatted_text = f"{self.bot_name}\n\n{text}"
        
        # Schedule the coroutine in the loop where the application is running
        asyncio.run_coroutine_threadsafe(
            self._async_send_response(chat_jid, formatted_text, original_message), 
            self._loop
        )

    async def _async_send_response(self, chat_jid: str, text: str, update: Optional[Update] = None):
        try:
            # Use the provided update for reply if available, else send fresh
            if update and hasattr(update, "message"):
                await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            else:
                await self.application.bot.send_message(chat_id=chat_jid, text=text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Telegram Send Error: {e}")

    def start(self) -> None:
        if not self.application:
            logger.error("Cannot start Telegram Channel: Application not initialized.")
            return
            
        logger.info("Starting Telegram Polling...")
        
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Tell user to message the bot if they haven't authorized yet
        if self.authorized_user_id is None:
            print("\n" + "┌" + "─"*58 + "┐")
            print("│" + " "*17 + "📢  ACTION REQUIRED  📢" + " "*18 + "│")
            print("├" + "─"*58 + "┤")
            print("│ Please send a message to your bot on Telegram now.       │")
            print("│ This will allow you to authorize your account.           │")
            print("└" + "─"*58 + "┘\n")

        self.application.run_polling(close_loop=True)

    def stop(self) -> None:
        logger.info("Stopping Telegram Channel...")
        if self._loop and self._loop.is_running():
            # PTB usually handles exit via signals, but we try a clean stop
            self._loop.call_soon_threadsafe(self._loop.stop)
