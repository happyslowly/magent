import httpx
from loguru import logger

from .message import Message, ModelMessage, ToolCallMessage


class OpenAIProvider:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key


class OpenAIModel:
    def __init__(self, model_name: str, provider: OpenAIProvider):
        self.model_name = model_name
        self.provider = provider
        self.chat_url = f"{self.provider.base_url}/chat/completions"

    async def invoke(self, messages: list[Message], **kwargs) -> Message | None:
        async with httpx.AsyncClient() as client:
            body = {
                "model": self.model_name,
                "messages": [m.model_dump() for m in messages],
                "tools": kwargs.get("tools", []),
            }
            logger.debug(body)
            try:
                response = await client.post(self.chat_url, json=body)
                response.raise_for_status()
                response_data = response.json()
                choice = response_data["choices"][0]
                reason = choice["finish_reason"]
                response_message = choice["message"]
                if reason == "tool_calls":
                    return ToolCallMessage(
                        content=response_message["content"],
                        tool_calls=response_message["tool_calls"],
                    )
                else:
                    return ModelMessage(content=response_message["content"])
            except Exception as e:
                logger.error("Model call failed: ", e)
