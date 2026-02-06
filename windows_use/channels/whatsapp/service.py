import os
import asyncio
import re
import logging
from pathlib import Path
from neonize.client import NewClient, DeviceProps
from neonize.events import (
    ConnectedEv,
    MessageEv,
    PairStatusEv,
    DisconnectedEv
)
from neonize.utils.jid import build_jid
from ..base import BaseChannel

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WhatsApp")

class WhatsAppChannel(BaseChannel):
    def __init__(self, session_dir: str = None, trigger_phrase: str = "@agent", 
                 bot_name: str = "🤖 *Windows Use*",
                 profile_name: str = "Windows Use",
                 profile_status: str = "Operating your Windows OS using AI"):
        self.trigger_phrase = trigger_phrase
        self.bot_name = bot_name
        self.profile_name = profile_name
        self.profile_status = profile_status
        self.trigger_regex = re.compile(f'^{re.escape(trigger_phrase)}\\b', re.IGNORECASE)
        
        # Setup Session Directory
        if session_dir:
            self.session_dir = Path(session_dir)
        else:
            user_home = Path(os.path.expanduser("~"))
            self.session_dir = user_home / ".windows-use" / "channels" / "whatsapp"
            
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / "whatsapp_session.sqlite"

        # Client Init
        self.device_props = DeviceProps(
            os="windows-use",
            platformType=DeviceProps.PlatformType.DESKTOP,
            requireFullSync=False
        )
        
        self.client = NewClient(str(self.session_file), props=self.device_props)
        self._loop = asyncio.get_event_loop() # Capture the main event loop
        self._msg_queue = asyncio.Queue()  # Always initialize the queue
        self._register_events()

    def _register_events(self):
        @self.client.event(ConnectedEv)
        def on_connected(c, event: ConnectedEv):
            logger.info("Connected to WhatsApp Server as 'windows-use'")
            self._on_init_tasks(c)

        @self.client.event(DisconnectedEv)
        def on_disconnected(c, event: DisconnectedEv):
            logger.warning(f"Disconnected: {event}")

        @self.client.event(PairStatusEv)
        def on_pair_status(c, event: PairStatusEv):
            logger.info(f"Pair Status: {event}")

        @self.client.event(MessageEv)
        def on_message(c, message: MessageEv):
            self._handle_message(c, message)

    def _on_init_tasks(self, c):
        try:
            me = c.get_me()
            self.owner_jid = build_jid(me.JID.User, me.JID.Server)
            
            # --- PROFILE SYNC ---
            print("\n" + "╔" + "═"*48 + "╗")
            print("║" + " "*17 + "📱  WHATSAPP READY" + " "*15 + "║")
            print("╠" + "═"*48 + "╣")
            print(f"║ Owner: {str(self.owner_jid):<41}║")
            print(f"║ Status: Online{'':<32}║")
            print("╚" + "═"*48 + "╝\n")
            
            # 1. Update Profile Information
            try:
                # Set Status Message
                c.set_status_message(self.profile_status)
                logger.info(f"Status set to: {self.profile_status}")
            except Exception as e:
                logger.debug(f"Could not set status: {e}")

            # 2. Update Profile Photo
            logo_path = Path("assets/logo.png")
            
            if logo_path.exists():
                try:
                    c.set_profile_photo(str(logo_path))
                    logger.info(f"Profile photo updated from {logo_path.name}")
                    
                    # Send Welcome Message to Owner
                    c.send_image(
                        self.owner_jid, 
                        str(logo_path), 
                        caption=f"*{self.profile_name}* is Ready.\n\nI am your Windows Use agent. You can now send commands directly to me in this chat."
                    )
                except Exception: 
                    pass 

            logger.info("WhatsApp Profile Ready.")
            
        except Exception as e:
            logger.error(f"Init Error: {e}")

    async def listen_for_messages(self):
        """
        Async generator that yields (chat_jid, text, media_path, original_msg) tuples.
        """
        while True:
            chat_jid, text, media_path, original_msg = await self._msg_queue.get()
            yield chat_jid, text, media_path, original_msg

    def send_response(self, chat_jid, text, original_message=None):
        try:
            # Prefix with persona to feel like a "second person"
            formatted_text = f"{self.bot_name}\n\n{text}"
            
            if original_message:
                self.client.reply_message(formatted_text, original_message)
            else:
                self.client.send_message(chat_jid, formatted_text)
        except Exception as e:
            logger.error(f"Send Error: {e}")

    def _handle_message(self, c, message: MessageEv):
        text = self._extract_text(message)
        has_audio = bool(message.Message.audioMessage)
        
        # Identify if bot was addressed (explicitly or via DM)
        is_text_triggered = bool(text and self.trigger_regex.match(text))
        is_dm = "s.whatsapp.net" in str(message.Info.MessageSource.Chat)
        
        # Skip if it's just a random group message or non-command audio
        if not is_text_triggered and not is_dm and not has_audio:
            return

        # Ensure owner is identified
        if not hasattr(self, "owner_jid"):
             try:
                 me = c.get_me()
                 self.owner_jid = build_jid(me.JID.User, me.JID.Server)
             except:
                 return

        # Security Check
        # Most reliable check: Is the message from the account owner?
        is_from_me = getattr(message.Info, "IsFromMe", False)
        
        source = message.Info.MessageSource
        sender_jid = source.Sender if (hasattr(source, "Sender") and source.Sender.User) else source.Chat
        
        # DEBUG SENDER INFO
        logger.info(f"--- MESSAGE DEBUG ---")
        logger.info(f"IsFromMe: {is_from_me}")
        logger.info(f"Owner JID: {self.owner_jid}")
        logger.info(f"Sender JID: {sender_jid}")
        logger.info(f"Chat JID: {source.Chat}")
        logger.info(f"----------------------")

        # Security Check (DISABLED FOR NOW)
        # clean_sender = sender_jid.User.split(':')[0].split('.')[0]
        # clean_owner = self.owner_jid.User.split(':')[0].split('.')[0]
        # if not is_from_me and clean_sender != clean_owner:
        #      if is_text_triggered or is_dm:
        #           logger.warning(f"Unauthorized access attempt from {sender_jid.User}")
        #           c.reply_message("your not my owner", message)
        #      return

        # Process Command
        cmd_content = ""
        if is_text_triggered:
            cmd_content = self.trigger_regex.sub('', text).strip()
        elif is_dm and has_audio:
             # In DMs, we treat any audio as a command (even without @agent)
            cmd_content = "(audio)"
        elif is_dm and text:
            # In DMs, we allow commands without the prefix to feel like a "second person" conversation.
            cmd_content = self.trigger_regex.sub('', text).strip() if is_text_triggered else text.strip()

        if not cmd_content:
             return

        logger.info(f"Command Queueing: {cmd_content}")

        media_path = None
        def is_valid_media(m):
             return m and hasattr(m, "url") and m.url

        downloadable_msg = None
        ext = ""
        media_type = None

        if is_valid_media(message.Message.imageMessage):
            downloadable_msg = message.Message 
            ext = ".jpg"
            media_type = "image"
        elif is_valid_media(message.Message.audioMessage):
            downloadable_msg = message.Message
            ext = ".ogg"
            media_type = "audio"
            
        if downloadable_msg:
             try:
                 logger.info(f"Downloading {media_type}...")
                 media_data = c.download_any(downloadable_msg)
                 if media_data:
                     import tempfile
                     fd, path = tempfile.mkstemp(suffix=ext)
                     with os.fdopen(fd, 'wb') as tmp:
                         tmp.write(media_data)
                     media_path = path
                     logger.info(f"Saved to {media_path}")
             except Exception as e:
                 logger.error(f"Media download failed: {e}")

        chat_jid = message.Info.MessageSource.Chat
        
        # Thread-safe queue put
        self._loop.call_soon_threadsafe(
            self._msg_queue.put_nowait, 
            (chat_jid, cmd_content, media_path, message)
        )

    def _extract_text(self, message: MessageEv):
        try:
            msg = message.Message
            if msg.conversation:
                return msg.conversation
            if msg.extendedTextMessage:
                return msg.extendedTextMessage.text
            if msg.imageMessage and msg.imageMessage.caption:
                return msg.imageMessage.caption
            if msg.videoMessage and msg.videoMessage.caption:
                return msg.videoMessage.caption
        except:
            pass
        return None

    async def _run_system_command(self, cmd, media_path=None):
        try:
            env_mods = os.environ.copy()
            if media_path:
                env_mods["WHATSAPP_IMAGE"] = media_path
                logger.debug(f"Executing command with image available at {media_path}")

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode().strip() or stderr.decode().strip()
            
            note = ""
            if media_path:
                 note = "\n(Image received and processed)"

            if not output:
                 return f"Done (No output){note}"
            return f"Output:\n{output}{note}"
        except Exception as e:
            return f"Execution Failed: {str(e)}"

    def start(self):
        print("\n" + "╔" + "═"*48 + "╗")
        print("║" + " "*14 + "WINDOWS USE: WHATSAPP SETUP" + " "*13 + "║")
        print("╠" + "═"*48 + "╣")
        print("║ 1. Ensure you have WhatsApp Web scanned.       ║")
        print("║ 2. The bot will automatically connect.         ║")
        print("║ 3. Session stored in .windows-use folder.      ║")
        print("╚" + "═"*48 + "╝")
        
        logger.info(f"Session File: {self.session_file}")
        try:
            self.client.connect()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("Stopping WhatsApp Service...")
        self.client.disconnect()
