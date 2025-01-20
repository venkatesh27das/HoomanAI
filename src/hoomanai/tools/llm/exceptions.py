class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMConnectionError(LLMError):
    """Raised when there's an error connecting to the LLM service."""
    pass

class LLMResponseError(LLMError):
    """Raised when there's an error in the LLM response."""
    pass