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
print(response)
# ModelMessage(role="The dice roll result is 4. You guessed 5, which was quite close! Would you like to play again?")
print(agent.get_all_messages())
# [SystemMessage(role='system', content='Play a number guessing game'),
#  HumanMessage(role='user', content='I guess 5'),
#  ToolCallMessage(role='assistant', content=None, tool_calls=[{'id': 'call_l1hkY7onoxdJGPfMzHByT8O9', 'type': 'function', 'function': {'name': 'roll_dice', 'arguments': '{}'}}]),
#  ToolMessage(role='tool', content='4', tool_call_id='call_l1hkY7onoxdJGPfMzHByT8O9'),
#  ModelMessage(role='assistant', content='The dice roll result is 4. You guessed 5, which was quite close! Would you like to play again?')]
```

