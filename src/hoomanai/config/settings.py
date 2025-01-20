from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AgentFrameworkConfig:
    """Configuration settings for the agent framework."""
    llm_settings: Dict[str, Any] = None
    memory_settings: Dict[str, Any] = None
    logging_settings: Dict[str, Any] = None

# Example configuration usage
default_config = AgentFrameworkConfig(
    llm_settings={
        "model_name": "llama-3.2-3b-instruct",
        "temperature": 0.7,
        "base_url": "http://localhost:1234/v1"
    },
    memory_settings={
        "memory_type": "short_term",
        "max_entries": 1000
    },
    logging_settings={
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
)
