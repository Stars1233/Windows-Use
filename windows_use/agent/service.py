from windows_use.messages import SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from windows_use.agent.tools import BUILTIN_TOOLS,EXPERIMENTAL_TOOLS
from windows_use.telemetry.views import AgentTelemetryEvent
from windows_use.agent.views import AgentResult,AgentState
from windows_use.telemetry.service import ProductTelemetry
from windows_use.agent.registry.service import Registry
from windows_use.agent.watchdog.service import WatchDog
from windows_use.agent.registry.views import ToolResult
from windows_use.agent.desktop.service import Desktop
from windows_use.agent.desktop.views import Browser
from windows_use.agent.prompt.service import Prompt
from windows_use.llms.base import BaseChatLLM
from windows_use.agent.base import BaseAgent
from windows_use.uia import Control
from contextlib import nullcontext
from rich.markdown import Markdown
from rich.console import Console
from typing import Literal
import logging
import time

logger = logging.getLogger("windows_use")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Agent(BaseAgent):
    def __init__(self,mode:Literal["flash","normal"]="normal",instructions:list[str]=[],browser:Browser=Browser.EDGE, use_annotation:bool=False, llm: BaseChatLLM=None,max_consecutive_failures:int=3,max_steps:int=25,use_vision:bool=False,auto_minimize:bool=False,experimental:bool=False):
        '''
        Initialize the Agent.

        The Agent is the core component that orchestrates interactions with the Windows GUI.
        It uses an LLM to process instructions, analyze the desktop state (via UI automation 
        and optionally vision), and execute tools to achieve the desired goals.

        Args:
            instructions (list[str]): A list of additional instructions or goals for the agent to execute.
            browser (Browser): The target web browser for web-related tasks. Defaults to Browser.EDGE.
            use_annotation (bool): Whether to overlay UI element annotations on screenshots before providing to the LLM. Defaults to False.
            llm (BaseChatLLM): The Large Language Model instance used for decision making.
            max_consecutive_failures (int): Maximum number of consecutive failures before giving up.
            max_steps (int): Maximum number of steps allowed in the agent's execution.
            use_vision (bool): Whether to provide screenshots to the LLM. Defaults to False.
            auto_minimize (bool): Whether to automatically minimize the current window before agent proceeds. Defaults to False.
        '''
        self.name='Windows Use'
        self.description='An agent that can interact with GUI elements on Windows OS'
        self.mode=mode
        self.registry = Registry(BUILTIN_TOOLS if experimental else BUILTIN_TOOLS+EXPERIMENTAL_TOOLS)
        self.instructions=instructions
        self.browser=browser
        self.max_steps=max_steps
        self.max_consecutive_failures=max_consecutive_failures
        self.auto_minimize=auto_minimize
        self.use_annotation=use_annotation
        if self.use_annotation and not use_vision:
            logger.warning("use_vision is set to True if use_annotation is True.")
        self.use_vision=True if use_annotation else use_vision
        self.state=AgentState(max_steps=max_steps,max_consecutive_failures=max_consecutive_failures)
        self.telemetry=ProductTelemetry()
        self.watchdog = WatchDog()
        self.desktop = Desktop()
        self.console=Console()
        self.prompt=Prompt()
        self.llm = llm
        self.step=0

    @property
    def system_message(self)->SystemMessage:
        content=self.prompt.system(
            mode=self.mode,
            desktop=self.desktop,
            browser=self.browser,
            max_steps=self.max_steps,
            instructions=self.instructions
        )
        return SystemMessage(content=content)

    @property
    def tools(self):
        return self.registry.get_tools()

    @property
    def state_message(self)->HumanMessage|ImageMessage:
        desktop_state=self.desktop.get_state(
            use_annotation=self.use_annotation,
            use_vision=self.use_vision
        )
        content=self.prompt.human(
            query=self.state.task,
            step=self.state.step,
            max_steps=self.state.max_steps,
            desktop_state=desktop_state
        )
        if self.use_vision and desktop_state.screenshot:
            image=desktop_state.screenshot
            return ImageMessage(image=image,content=content)
        return HumanMessage(content=content)

    def reason(self)->ToolMessage|AIMessage:
        max_attempts = self.max_consecutive_failures
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                llm_response = self.llm.invoke(messages=self.state.messages, tools=self.tools)
                return llm_response.content
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:  # Don't sleep after the last attempt
                    wait_time = 2 ** (attempt + 1)
                    logger.error(
                        f"Failed to get response from {self.llm.provider} for {self.llm.model_name}.\n"
                        f"Retrying...({attempt + 1}/{max_attempts})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to get response from {self.llm.provider} for {self.llm.model_name}.\n"
                        f"All {max_attempts} attempts exhausted."
                    )

        if last_error:
            raise last_error
        else:
            raise RuntimeError("Failed to get response from LLM: Unknown error")

    def act(self,tool_name:str,tool_params:dict)->ToolResult:
        if 'thought' in tool_params and tool_params['thought'] is not None:
            logger.info(f"[Agent] 🧠 Thinking: {tool_params['thought']}")
        formatted_params=[
            f"{key}={value}" for key, value in tool_params.items()
            if key not in ['thought', 'evaluate']
        ]
        if not tool_name.startswith("done_tool"):
            logger.info(f"[Agent] 🛠️ Tool Call: {tool_name}({', '.join(formatted_params)})")
        tool_result=self.registry.execute(tool_name=tool_name,tool_params=tool_params,desktop=self.desktop)
        if tool_result.is_success:
            if not tool_name.startswith("done_tool"):
                logger.info(f"[Agent] 📃 Tool Result: {tool_result.content}")
        else:
            if not tool_name.startswith("done_tool"):
                logger.error(f"[Agent] ❌ Tool Error: {tool_result.error}")
        return tool_result

    def answer(self,content:str)->str:
        logger.info(f"[Agent] 📜 Final Answer: {content}")
        return content

    def loop(self):
        self.state.messages.insert(0,self.system_message)
        for step in range(self.max_steps):
            self.state.step=step
            self.state.messages.append(self.state_message)
            message=self.reason()
            if isinstance(message,ToolMessage):
                self.state.messages.pop() # Remove the state message from the messages list
                tool_name=message.name
                tool_params=message.params
                tool_result=self.act(
                    tool_name=tool_name,
                    tool_params=tool_params
                )
                content=tool_result.content if tool_result.is_success else tool_result.error
                message.content=content
                if tool_name.startswith("done_tool"):
                    return self.answer(content=content)
                self.state.messages.append(message)
            if isinstance(message,AIMessage):
                self.state.messages.append(message)
                content=message.content
                return self.answer(content=content)

    def invoke(self,query:str):
        self.state.task=query
        with (self.desktop.auto_minimize() if self.auto_minimize else nullcontext()):
            self.watchdog.set_focus_callback(self.desktop.tree.on_focus_change)
            with self.watchdog:
                content=self.loop()
        return AgentResult(
            content=content,
        )