from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Message:
    """Represents a single message in the conversation with the LLM."""
    role: str  # "system", "user", or "assistant"
    content: str

@dataclass
class LLMRequest:
    """Represents a request to the LLM API."""
    model: str
    messages: List[Message]
    temperature: float
    max_tokens: int
    stream: bool = False

@dataclass
class LLMResponse:
    """Represents a response from the LLM API."""
    content: str
    raw_response: Dict[str, Any]