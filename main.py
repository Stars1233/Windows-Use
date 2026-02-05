# from windows_use.llms.google import ChatGoogle
from windows_use.llms.anthropic import ChatAnthropic
from windows_use.llms.ollama import ChatOllama
from windows_use.llms.mistral import ChatMistral
from windows_use.llms.azure_openai import ChatAzureOpenAI
from windows_use.llms.open_router import ChatOpenRouter
from windows_use.llms.groq import ChatGroq
from windows_use.agent import Agent, Browser
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    # llm=ChatMistral(model='magistral-small-latest',temperature=0.7)
    # llm=ChatGoogle(model="gemini-2.5-flash-lite",temperature=0.7) #Google wrapper is under developement
    # llm=ChatOpenRouter(model="nvidia/nemotron-3-nano-30b-a3b:free",temperature=0.2)
    # llm=ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)
    # llm=ChatOllama(model="qwen3-vl:4b")
    # llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    agent = Agent(mode='normal',llm=llm, use_vision=True, browser=Browser.EDGE,auto_minimize=False)
    agent.invoke(query=input("Enter a query: "))

if __name__ == "__main__":
    main()