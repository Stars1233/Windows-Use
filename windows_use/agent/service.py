from windows_use.messages import SystemMessage, HumanMessage, AIMessage, ImageMessage, ToolMessage
from windows_use.agent.tools import BUILTIN_TOOLS, EXPERIMENTAL_TOOLS
from windows_use.telemetry.views import AgentTelemetryEvent
from windows_use.agent.views import AgentResult, AgentState
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
    def __init__(
        self,
        mode: Literal["flash", "normal"] = "normal",
        instructions: list[str] | None = None,
        browser: Browser = Browser.EDGE,
        use_annotation: bool = False,
        use_accessibility: bool = True,
        llm: BaseChatLLM = None,
        max_consecutive_failures: int = 3,
        max_steps: int = 25,
        use_vision: bool = False,
        auto_minimize: bool = False,
        experimental: bool = False,
    ):
        """Initialize the Agent.

        The Agent is the core component that orchestrates interactions with the Windows GUI.
        It uses an LLM to process instructions, analyze the desktop state (via UI automation
        and optionally vision), and execute tools to achieve the desired goals.

        Args:
            mode: Agent mode - "flash" for lightweight prompts, "normal" for full prompts.
            instructions: A list of additional instructions or goals for the agent to execute.
            browser: The target web browser for web-related tasks. Defaults to Browser.EDGE.
            use_annotation: Whether to overlay UI element annotations on screenshots before
                providing to the LLM. Defaults to False.
            use_accessibility: Whether to use the accessibility tree. Defaults to True.
            llm: The Large Language Model instance used for decision making.
            max_consecutive_failures: Maximum number of consecutive failures before giving up.
            max_steps: Maximum number of steps allowed in the agent's execution.
            use_vision: Whether to provide screenshots to the LLM. Defaults to False.
            auto_minimize: Whether to automatically minimize the current window before agent
                proceeds. Defaults to False.
            experimental: Whether to include experimental tools. Defaults to False.
        """
        self.name = "Windows Use"
        self.description = "An agent that can interact with GUI elements on Windows OS"
        self.mode = mode
        self.registry = Registry(
            BUILTIN_TOOLS + EXPERIMENTAL_TOOLS if experimental else BUILTIN_TOOLS
        )
        self.instructions = instructions or []
        self.browser = browser
        self.max_steps = max_steps
        self.max_consecutive_failures = max_consecutive_failures
        self.auto_minimize = auto_minimize
        if use_annotation and not use_vision:
            logger.warning("use_vision is set to True if use_annotation is True.")
        if use_annotation and not use_accessibility:
            logger.warning("use_accessibility is set to True if use_annotation is True.")
        self.desktop = Desktop(
            use_vision=True if use_annotation else use_vision,
            use_annotation=use_annotation,
            use_accessibility=True if use_annotation else use_accessibility,
        )
        self.state = AgentState(
            max_steps=max_steps, max_consecutive_failures=max_consecutive_failures
        )
        self.telemetry = ProductTelemetry()
        self.watchdog = WatchDog()
        self.console = Console()
        self.prompt = Prompt()
        self.llm = llm
        self.step = 0

    @property
    def system_message(self) -> SystemMessage:
        content = self.prompt.system(
            mode=self.mode,
            desktop=self.desktop,
            browser=self.browser,
            max_steps=self.max_steps,
            instructions=self.instructions,
        )
        return SystemMessage(content=content)

    @property
    def tools(self):
        return self.registry.get_tools()

    @property
    def state_message(self) -> HumanMessage | ImageMessage:
        desktop_state = self.desktop.get_state()
        content = self.prompt.human(
            query=self.state.task,
            step=self.state.step,
            max_steps=self.state.max_steps,
            desktop=self.desktop,
        )
        if self.desktop.use_vision and desktop_state.screenshot:
            image = desktop_state.screenshot
            return ImageMessage(image=image, content=content)
        return HumanMessage(content=content)

    def reason(self) -> ToolMessage | AIMessage:
        """Call the LLM and return the response message.

        Retries up to max_consecutive_failures times with exponential backoff on failure.

        Returns:
            The LLM response as a ToolMessage or AIMessage.

        Raises:
            ValueError: If the LLM returns an unexpected response type.
            Exception: If all retry attempts are exhausted.
        """
        max_attempts = self.max_consecutive_failures
        last_error = None

        for attempt in range(max_attempts):
            try:
                llm_response = self.llm.invoke(messages=self.state.messages, tools=self.tools)
                content = llm_response.content

                if content is None:
                    raise ValueError(
                        f"LLM returned None content (provider: {self.llm.provider}, "
                        f"model: {self.llm.model_name})"
                    )

                if not isinstance(content, (ToolMessage, AIMessage)):
                    raise ValueError(
                        f"LLM returned unexpected content type: {type(content).__name__}. "
                        f"Expected ToolMessage or AIMessage."
                    )

                return content
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.error(
                        f"Failed to get response from {self.llm.provider} "
                        f"for {self.llm.model_name}.\n"
                        f"Retrying...({attempt + 1}/{max_attempts})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to get response from {self.llm.provider} "
                        f"for {self.llm.model_name}.\n"
                        f"All {max_attempts} attempts exhausted."
                    )

        raise last_error

    def act(self, tool_name: str, tool_params: dict) -> ToolResult:
        tool_result = self.registry.execute(
            tool_name=tool_name, tool_params=tool_params, desktop=self.desktop
        )
        return tool_result

    def result(self, content: str, is_done: bool = True) -> AgentResult:
        return AgentResult(content=content, is_done=is_done)

    def loop(self) -> AgentResult:
        """Run the main agent loop.

        Iterates up to max_steps, calling the LLM and executing tools each step.
        Tracks consecutive tool failures and aborts if the threshold is reached.

        Returns:
            AgentResult with the final answer or an error/timeout indication.
        """
        self.state.messages.insert(0, self.system_message)
        consecutive_failures = 0

        for step in range(self.max_steps):
            self.state.step = step
            self.state.messages.append(self.state_message)

            try:
                message = self.reason()
            except Exception as e:
                logger.error(f"[Agent] Step {step + 1}/{self.max_steps} - Reason failed: {e}")
                return AgentResult(
                    is_done=False,
                    error=f"Agent failed after exhausting retries: {e}",
                )

            if isinstance(message, ToolMessage):
                self.state.messages.pop()  # Remove the state message from the messages list

                tool_name = message.name
                tool_params = message.params

                evaluate = tool_params.get('evaluate', 'neutral')
                logger.info(f"[Agent] 📊 Evaluate: {evaluate}")
                logger.info(f"[Agent] 🧠 Thinking: {tool_params.get('thought', '')}")

                if tool_name != "done_tool":
                    formatted_params = [
                        f"{key}={value}"
                        for key, value in tool_params.items()
                        if key not in ["thought", "evaluate"]
                    ]
                    logger.info(
                        f"[Agent] 🛠️ Tool Call: {tool_name}({', '.join(formatted_params)})"
                    )

                tool_result = self.act(tool_name=tool_name, tool_params=tool_params)
                content = tool_result.content if tool_result.is_success else tool_result.error
                message.content = content
                self.state.messages.append(message)

                # Track consecutive failures
                if not tool_result.is_success:
                    consecutive_failures += 1
                    logger.warning(
                        f"[Agent] Tool '{tool_name}' failed "
                        f"({consecutive_failures}/{self.max_consecutive_failures}): {content}"
                    )
                    if consecutive_failures >= self.max_consecutive_failures:
                        logger.error(
                            f"[Agent] Aborting: {self.max_consecutive_failures} "
                            f"consecutive tool failures reached."
                        )
                        return AgentResult(
                            is_done=False,
                            error=(
                                f"Agent aborted after {self.max_consecutive_failures} "
                                f"consecutive tool failures. Last error: {content}"
                            ),
                        )
                else:
                    consecutive_failures = 0

                if tool_name != "done_tool":
                    logger.info(f"[Agent] 📝 Tool Result: {content}")

                if tool_name == "done_tool":
                    logger.info(f"[Agent] 📝 Final Answer: {content}")
                    return self.result(content=content, is_done=True)

            if isinstance(message, AIMessage):
                self.state.messages.append(message)
                content = message.content
                return self.result(content=content, is_done=True)

        logger.warning(
            f"[Agent] Max steps ({self.max_steps}) reached without completing the task."
        )
        return AgentResult(
            is_done=False,
            error=f"Agent reached the maximum number of steps ({self.max_steps}) without completing.",
        )

    def invoke(self, query: str) -> AgentResult:
        self.state.task = query
        with self.desktop.auto_minimize() if self.auto_minimize else nullcontext():
            self.watchdog.set_focus_callback(self.desktop.tree.on_focus_change)
            with self.watchdog:
                result = self.loop()
        return result