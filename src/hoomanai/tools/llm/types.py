from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

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
    """
    Represents a response from an LLM service.
    
    Attributes:
        content: The actual text response from the LLM
        model: The model used to generate the response
        created_at: Timestamp of response creation
        token_usage: Dictionary containing token usage information
        finish_reason: Reason why the LLM stopped generating
        context_window: Size of context window used
        response_metadata: Additional metadata from the LLM service
    """
    content: str
    model: str
    created_at: datetime
    token_usage: Dict[str, int]  # e.g., {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
    finish_reason: Optional[str] = None  # e.g., 'stop', 'length', 'content_filter'
    context_window: Optional[int] = None  # Size of context window used
    response_metadata: Optional[Dict[str, Any]] = None  # Any additional metadata from the LLM service