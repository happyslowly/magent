# magent

A minimal AI agent framework for tool-calling with OpenAI-compatible APIs.

## Features

- **Tool calling**: Automatic function execution with OpenAI-style tool calling
- **Multi-turn conversations**: Persistent thread management
- **Error handling**: Graceful failure recovery
- **Type safety**: Full Pydantic model validation

## Quick Start

```python
from magent.agent import Agent
from magent.model import OpenAIModel, OpenAIProvider

def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    import random
    return str(random.randint(1, 6))

# Setup model and agent
model = OpenAIModel(
    "gpt-4",
    OpenAIProvider(base_url="https://api.openai.com/v1", api_key="your-key")
)

agent = Agent(
    model=model,
    system_prompt="Play a number guessing game"
    tools=[roll_dice]
)

# Use the agent
response = await agent.invoke("I guess 5")
# response:
# ModelMessage(role="assistant" content="Sorry, but your guess of 5 is incorrect. The dice rolled a 3. Would you like to try again?\n")
```

