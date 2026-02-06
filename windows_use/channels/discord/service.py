import os
import asyncio
import logging
import json
import discord
from typing import AsyncGenerator, Tuple, Any, Optional
from pathlib import Path
from ..base import BaseChannel

logger = logging.getLogger("Discord")

class DiscordChannel(BaseChannel):
    def __init__(self, token: str = None, bot_name: str = "🤖 *Windows Use*"):
        self.token = token
        self.bot_name = bot_name
        self._msg_queue = asyncio.Queue()
        self._loop = None
        
        # Setup Session Directory
        user_home = Path(os.path.expanduser("~"))
        self.session_dir = user_home / ".windows-use" / "channels" / "discord"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.session_dir / "config.json"
        
        self.authorized_user_id = None
        self._load_config()
        
        # Discord Client Setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        self.client = discord.Client(intents=intents)
        self._setup_event_handlers()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.token = self.token or config.get("token")
                    self.authorized_user_id = config.get("authorized_user_id")
                    logger.info("Loaded Discord config.")
            except Exception as e:
                logger.error(f"Failed to load Discord config: {e}")

    def _save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "token": self.token,
                    "authorized_user_id": self.authorized_user_id
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save Discord config: {e}")

    def _ensure_token(self):
        if not self.token:
            self.token = os.getenv("DISCORD_BOT_TOKEN")
            
        if not self.token:
            print("\n" + "╔" + "═"*48 + "╗")
            print("║" + " "*14 + "WINDOWS USE: DISCORD SETUP" + " "*13 + "║")
            print("╠" + "═"*48 + "╣")
            print("║ 1. Go to Discord Developer Portal.             ║")
            print("║ 2. Create an App -> Bot.                       ║")
            print("║ 3. Reset/Copy the Bot Token.                   ║")
            print("║ 4. Enable 'Message Content Intent'.            ║")
            print("╚" + "═"*48 + "╝")
            self.token = input("\n ➤ Enter your Discord Bot Token: ").strip()
            if not self.token:
                logger.error("No token provided. Discord channel will not function.")
                return False
            self._save_config()
        return True

    def _setup_event_handlers(self):
        @self.client.event
        async def on_ready():
            logger.info(f"Discord connected as {self.client.user}")
            if self.authorized_user_id is None:
                print("\n" + "┌" + "─"*58 + "┐")
                print("│" + " "*17 + "📢  ACTION REQUIRED  📢" + " "*18 + "│")
                print("├" + "─"*58 + "┤")
                print("│ Please DM your bot or mention it in a server.            │")
                print("│ This will allow you to authorize your account.           │")
                print("└" + "─"*58 + "┘\n")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            # Determine if it's a command (DM or mention)
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = self.client.user in message.mentions
            
            if not is_dm and not is_mentioned:
                return

            # 1. Check Authorization
            if self.authorized_user_id is None:
                print("\n" + "╔" + "═"*48 + "╗")
                print("║" + " "*14 + "🔐  AUTHORIZATION REQUEST" + " "*13 + "║")
                print("╠" + "═"*48 + "╣")
                print(f"║ User: {message.author.name:<41}║")
                print(f"║ ID: {message.author.id:<43}║")
                print("╚" + "═"*48 + "╝")
                
                choice = await asyncio.to_thread(input, "\n ➤ Authorize this user to control your PC? (y/n): ")
                
                if choice.lower() == 'y':
                    self.authorized_user_id = message.author.id
                    self._save_config()
                    await message.reply(f"✅ *Authorization Successful!*\n\nWelcome {message.author.name}. I am now your Windows Agent.")
                else:
                    await message.reply("❌ You are not authorized.")
                    return

            # 2. Verify identity
            if message.author.id != self.authorized_user_id:
                return

            # 3. Process Content
            text = message.clean_content.replace(f'@{self.client.user.name}', '').strip()
            media_path = None
            
            if message.attachments:
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                        logger.info(f"Downloading Discord attachment: {attachment.filename}")
                        import tempfile
                        suffix = Path(attachment.filename).suffix
                        fd, path = tempfile.mkstemp(suffix=suffix)
                        os.close(fd)
                        await attachment.save(path)
                        media_path = path
                        break # Only take the first image for now
            
            # 4. Queue for Gateway
            if text or media_path:
                await self._msg_queue.put((message.channel.id, text, media_path, message))

    async def listen_for_messages(self) -> AsyncGenerator[Tuple[str, str, Optional[str], Any], None]:
        while True:
            chat_id, text, media_path, original_msg = await self._msg_queue.get()
            yield str(chat_id), text, media_path, original_msg

    def send_response(self, chat_jid: str, text: str, original_message: Optional[Any] = None) -> None:
        if not self._loop:
            return
            
        formatted_text = f"{self.bot_name}\n\n{text}"
        
        # Split text if it exceeds Discord's 2000 character limit
        chunks = [formatted_text[i:i+1900] for i in range(0, len(formatted_text), 1900)]
        
        for chunk in chunks:
            asyncio.run_coroutine_threadsafe(
                self._async_send_response(chat_jid, chunk, original_message),
                self._loop
            )

    async def _async_send_response(self, chat_jid: str, text: str, original_msg: Optional[discord.Message] = None):
        try:
            if original_msg:
                await original_msg.reply(text)
            else:
                channel = self.client.get_channel(int(chat_jid))
                if not channel:
                    channel = await self.client.fetch_channel(int(chat_jid))
                if channel:
                    await channel.send(text)
        except Exception as e:
            logger.error(f"Discord Send Error: {e}")

    def start(self) -> None:
        if not self._ensure_token():
            return
            
        self._loop = asyncio.get_event_loop()
        
        async def run_discord():
            try:
                async with self.client:
                    await self.client.start(self.token)
            except Exception as e:
                logger.error(f"Discord start error: {e}")

        asyncio.create_task(run_discord())

    def stop(self) -> None:
        logger.info("Stopping Discord Channel...")
        if self._loop:
            asyncio.run_coroutine_threadsafe(self.client.close(), self._loop)
