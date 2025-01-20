from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

@dataclass
class LLMRequest:
    model: str
    messages: List[Message]
    temperature: float
    max_tokens: int
    stream: bool = False

@dataclass
class LLMResponse:
    content: str
    raw_response: Dict[str, Any]