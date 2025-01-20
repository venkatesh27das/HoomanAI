from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""
    base_url: str = "http://localhost:1234/v1"
    model_name: str = "llama-3.2-3b-instruct"
    temperature: float = 0.7
    max_tokens: int = -1
    default_system_message: str = "You are a helpful AI assistant."
