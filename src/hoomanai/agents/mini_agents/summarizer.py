from typing import Any, Dict, Optional
from datetime import datetime
from uuid import UUID

from hoomanai.core.mini_agent import MiniAgent
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError
from hoomanai.core.types import Task

class SummarizerAgent(MiniAgent):
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(config=llm_config)
        
    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the summarization task.
        
        Args:
            task: Task object containing the text to be summarized
            
        Returns:
            Dictionary containing the summary and metadata
            
        Raises:
            ValueError: If input text is missing or invalid
            LLMConnectionError: If there's an error connecting to the LLM service
            LLMResponseError: If there's an error in the model's response
        """
        try:
            # Extract text from task context
            if not task.context or 'text' not in task.context:
                raise ValueError("No text provided for summarization")
                
            text = task.context['text']
            max_length = task.context.get('max_length', 500)
            style = task.context.get('style', 'concise')
            
            # Create summarization prompt
            prompt = self._create_summary_prompt(text, max_length, style)
            
            # Get summary from LLM
            response = self.llm_client.complete(
                prompt=prompt,
                system_message=self._get_system_message(style)
            )
            
            # Process and structure the response
            return {
                'summary': response.content,
                'metadata': {
                    'original_length': len(text),
                    'summary_length': len(response.content),
                    'style': style,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except (LLMConnectionError, LLMResponseError) as e:
            raise Exception(f"Summarization failed: {str(e)}")

    def _create_summary_prompt(self, text: str, max_length: int, style: str) -> str:
        """Create the prompt for the summarization task."""
        return f"""
        Please summarize the following text in a {style} style, keeping the summary under {max_length} characters:

        {text}

        Please provide a coherent and well-structured summary that captures the main points while maintaining readability.
        """

    def _get_system_message(self, style: str) -> str:
        """Get the appropriate system message based on summarization style."""
        style_messages = {
            'concise': "You are a precise summarizer. Focus on extracting key points in a clear, concise manner.",
            'detailed': "You are a detailed summarizer. Provide comprehensive summaries while maintaining clarity.",
            'academic': "You are an academic summarizer. Use formal language and maintain academic rigor.",
        }
        return style_messages.get(style, style_messages['concise'])