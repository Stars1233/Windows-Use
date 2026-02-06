from windows_use.agent.service import Agent, Browser, AgentResult
from windows_use.channels.whatsapp import WhatsAppChannel
from windows_use.channels.telegram import TelegramChannel
from windows_use.channels.signal import SignalChannel
from windows_use.channels.discord import DiscordChannel
from windows_use.channels.slack import SlackChannel
from windows_use.llms.azure_openai import ChatAzureOpenAI
from windows_use.llms.open_router import ChatOpenRouter
from windows_use.llms.anthropic import ChatAnthropic
from windows_use.llms.mistral import ChatMistral
from windows_use.llms.google import ChatGoogle
from windows_use.llms.ollama import ChatOllama
from windows_use.llms.groq import ChatGroq
from windows_use.gateway import Gateway
from dotenv import load_dotenv
import asyncio
import logging
import os

# Explicitly load .env to avoid stack frame issues on some Windows environments
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Assistant")

async def main():
    # 1. Initialize the LLM
    # llm=ChatMistral(model='magistral-small-latest',temperature=0.7)
    llm=ChatGoogle(model="gemini-2.5-flash-lite",thinking_budget=0, temperature=0.7)
    # llm=ChatOpenRouter(model="nvidia/nemotron-3-nano-30b-a3b:free",temperature=0.2)
    # llm=ChatAnthropic(model="claude-sonnet-4-5", temperature=0.7,max_tokens=1000)
    # llm=ChatOllama(model="qwen3-vl:4b",temperature=0.2)
    # llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",temperature=0.7)
    
    # 2. Initialize the Windows Use Agent (The Brain)
    agent = Agent(llm=llm, browser=Browser.EDGE, use_vision=False)
    
    # 3. Initialize the Central Gateway
    # Passing the Agent instance directly to the Gateway
    gateway = Gateway(agent=agent)
    
    # 4. Add the Channels
    # whatsapp_channel=WhatsAppChannel()
    # gateway.add_channel(whatsapp_channel)
    
    # telegram_channel=TelegramChannel()
    # gateway.add_channel(telegram_channel)
    
    # signal_channel = SignalChannel()
    # gateway.add_channel(signal_channel)
    
    discord_channel = DiscordChannel()
    gateway.add_channel(discord_channel)
    
    # slack_channel = SlackChannel()
    # gateway.add_channel(slack_channel)
    
    # 5. Run the System
    try:
        await gateway.start()
    except KeyboardInterrupt:
        gateway.stop()

if __name__ == "__main__":
    asyncio.run(main())
