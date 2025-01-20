# HoomanAI: Extensible AI Agent Framework

HoomanAI is a powerful and flexible framework for building and orchestrating AI agents. It provides a structured way to create both simple (mini) agents and complex (compound) agents that can work together to accomplish sophisticated tasks.

## Installation

```bash
pip install hoomanai
```

## Quick Start

```python
from hoomanai.core import MasterAgent
from hoomanai.agents.mini_agents import SummarizerAgent
from hoomanai.core.types import UserContext

# Initialize agents
master = MasterAgent()
summarizer = SummarizerAgent("summarizer")
master.register_mini_agent("summarizer", summarizer)

# Create context and execute
context = UserContext(query="Summarize this text", additional_context={"text": "..."})
plan = master.create_execution_plan(context)
results = master.execute_plan(plan)
```

## Key Features

- **Flexible Agent Architecture**: Build both simple and complex agents
- **Task Orchestration**: Automatically create and execute multi-step plans
- **Built-in Agents**: Comes with several pre-built agents for common tasks
- **Extensible**: Easily create custom agents for specific use cases
- **Type Safety**: Full type hints and dataclass-based interfaces

## Creating Custom Agents

### Mini Agents

```python
from hoomanai.core import MiniAgent
from hoomanai.core.types import Task, AgentResponse

class MyCustomAgent(MiniAgent):
    def process(self, task: Task) -> AgentResponse:
        # Your processing logic here
        return AgentResponse(success=True, data=result)
```

### Compound Agents

```python
from hoomanai.core import CompoundAgent
from hoomanai.core.types import Task, AgentResponse

class MyCompoundAgent(CompoundAgent):
    def plan(self, task: Task) -> List[Task]:
        # Your planning logic here
        return sub_tasks
```

## Configuration

The framework can be configured using environment variables or a config file:

```env
HOOMANAI_LLM_PROVIDER=openai
HOOMANAI_LLM_MODEL=gpt-4
HOOMANAI_API_KEY=your-api-key
```

## Documentation

For full documentation, visit [docs.hoomanai.com](https://docs.hoomanai.com)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.