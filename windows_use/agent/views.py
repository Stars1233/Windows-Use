from windows_use.agent.desktop.views import DesktopState
from windows_use.messages import BaseMessage
from dataclasses import dataclass,field
from typing import Any

@dataclass
class AgentResult:
    is_done:bool=False
    content:str|None=None
    error:str|None=None

@dataclass
class AgentState:
    task:str=''
    messages:list[BaseMessage]=field(default_factory=list)
    error_messages:list[BaseMessage]=field(default_factory=list)
    desktop:DesktopState=None
    step:int=0
    max_steps:int=25
    max_consecutive_failures:int=3
