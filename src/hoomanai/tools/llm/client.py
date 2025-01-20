import requests
import json
from typing import Optional, Dict, Any, List
from .config import LLMConfig
from .exceptions import LLMConnectionError, LLMResponseError
from .types import Message, LLMRequest, LLMResponse

class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.headers = {
            "Content-Type": "application/json",
        }

    def _build_messages(self, prompt: str, system_message: Optional[str] = None) -> List[Message]:
        """Build messages list for the API request."""
        messages = [
            Message(
                role="system",
                content=system_message or self.config.default_system_message
            ),
            Message(role="user", content=prompt)
        ]
        return messages

    def _build_request(self, messages: List[Message], stream: bool = False) -> LLMRequest:
        """Build the request object for the API call."""
        return LLMRequest(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream
        )

    def _make_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Make the actual API request."""
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps({
                    "model": request.model,
                    "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream
                })
            )
            
            if response.status_code != 200:
                raise LLMConnectionError(f"Error {response.status_code}: {response.text}")
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"Failed to connect to LLM service: {str(e)}")

    def _parse_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """Parse the API response into a structured format."""
        if 'choices' not in response_data or not response_data['choices']:
            raise LLMResponseError("No valid response received from the model")
            
        content = response_data['choices'][0]['message']['content']
        return LLMResponse(content=content, raw_response=response_data)

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Main method to get completions from the LLM.
        
        Args:
            prompt: The input prompt for the model
            system_message: Optional system message to override default
            stream: Whether to stream the response
            
        Returns:
            LLMResponse object containing the completion
            
        Raises:
            LLMConnectionError: If there's an error connecting to the service
            LLMResponseError: If there's an error in the model's response
        """
        messages = self._build_messages(prompt, system_message)
        request = self._build_request(messages, stream)
        response_data = self._make_request(request)
        return self._parse_response(response_data)
