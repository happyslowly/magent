from typing import Literal

from pydantic import BaseModel

Role = Literal["user", "assistant", "tool", "system"]


class Message(BaseModel):
    role: Role
    content: str | None = None


class HumanMessage(Message):
    role: Role = "user"


class SystemMessage(Message):
    role: Role = "system"


class ModelMessage(Message):
    role: Role = "assistant"


class ToolMessage(Message):
    role: Role = "tool"
    tool_call_id: str


class ToolCallMessage(Message):
    role: Role = "assistant"
    tool_calls: list
