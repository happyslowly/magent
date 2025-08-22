import inspect
import json as _json
from collections import defaultdict
from typing import Callable, Literal, get_type_hints
from uuid import UUID

from loguru import logger

from .message import HumanMessage, Message, SystemMessage, ToolCallMessage, ToolMessage
from .model import OpenAIModel


class Agent:
    def __init__(
        self,
        model: OpenAIModel,
        system_prompt: str | None = None,
        tools: list[Callable] = [],
    ):
        self.model = model
        self.system_message = (
            SystemMessage(content=system_prompt) if system_prompt else None
        )
        self.threads = defaultdict(list)
        self._default_thread_id: Literal["default"] = "default"
        self.tools = tools
        self._tool_map = {t.__name__: t for t in self.tools}

    async def invoke(self, prompt: str, thread_id: UUID | None = None):
        input_messages: list[Message] = (
            [self.system_message] if self.system_message else []
        )
        human_message = HumanMessage(content=prompt)
        if thread_id and thread_id in self.threads:
            input_messages.extend(self.threads[thread_id])
        input_messages.append(human_message)
        tool_schemas = self._get_tool_schemas()

        while True:
            output_message = await self.model.invoke(input_messages, tools=tool_schemas)
            if not output_message:
                break

            if isinstance(output_message, ToolCallMessage):
                tool_messages = self._handle_tool_calls(output_message.tool_calls)
                input_messages.append(output_message)
                input_messages.extend(tool_messages)
            else:
                break

        if output_message:
            self._save_messages(
                thread_id or self._default_thread_id,
                input_messages[1:] + [output_message],
            )

        return output_message

    def _save_messages(
        self, thread_id: UUID | Literal["default"], messages: list[Message]
    ):
        self.threads[thread_id].extend(messages)

    def get_all_messages(self, thread_id: UUID | None = None, json: bool = False):
        messages = self._get_all_messages(thread_id or self._default_thread_id)
        if json:
            return _json.dumps([m.model_dump() for m in messages])

    def _get_all_messages(self, thread_id: UUID | Literal["default"]):
        history = self.threads.get(thread_id)
        return (
            ([self.system_message] if self.system_message else []) + history
            if history
            else []
        )

    def _handle_tool_calls(self, tool_calls: list):
        tool_results = []
        for tool_call in tool_calls:
            fn = self._tool_map.get(tool_call["function"]["name"])
            args = _json.loads(tool_call["function"]["arguments"])
            if fn:
                try:
                    result = fn(**args) if args else fn()
                    tool_results.append(
                        ToolMessage(content=result, tool_call_id=tool_call["id"])
                    )
                except Exception as e:
                    logger.error(f"Tool call failed: `{fn.__name__}`, `{args}`", e)
        return tool_results

    def _get_tool_schemas(self):
        schemas = []
        for tool in self.tools:
            sig = inspect.signature(tool)
            type_hints = get_type_hints(tool)

            doc = tool.__doc__ or ""
            lines = [line.strip() for line in doc.split("\n") if line.strip()]

            description = lines[0] if lines else tool.__name__

            properties = {}
            required = []

            in_args = False
            for line in lines:
                if line.startswith("Args:"):
                    in_args = True
                elif line.startswith("Returns:"):
                    break
                elif in_args and ":" in line:
                    param_name = line.split(":")[0].strip()
                    param_desc = line.split(":", 1)[1].strip()

                    param = sig.parameters.get(param_name)
                    if param:
                        properties[param_name] = {
                            "type": self._get_tool_param_type(param_name, type_hints),
                            "description": param_desc,
                        }

                        if param.default == inspect.Parameter.empty:
                            required.append(param_name)

            schema = {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            schemas.append(schema)
        return schemas

    @staticmethod
    def _get_tool_param_type(param_name, type_hints):
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(type_hints.get(param_name, str))
