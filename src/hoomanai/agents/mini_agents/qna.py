from typing import Any, Dict, Optional, List
from datetime import datetime

from hoomanai.core.mini_agent import MiniAgent
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError
from hoomanai.core.types import Task

class QnAAgent(MiniAgent):
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(config=llm_config)
        
    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the question-answering task.
        
        Args:
            task: Task object containing the question and context
            
        Returns:
            Dictionary containing the answer and confidence metrics
            
        Raises:
            ValueError: If question or context is missing
            LLMConnectionError: If there's an error connecting to the LLM service
            LLMResponseError: If there's an error in the model's response
        """
        try:
            # Validate input
            if not task.context or 'question' not in task.context:
                raise ValueError("No question provided")
                
            question = task.context['question']
            context = task.context.get('context', '')
            
            # Create QnA prompt
            prompt = self._create_qa_prompt(question, context)
            
            # Get answer from LLM
            response = self.llm_client.complete(
                prompt=prompt,
                system_message=self._get_system_message()
            )
            
            # Parse structured response
            answer_data = self._parse_response(response.content)
            
            return {
                'answer': answer_data['answer'],
                'confidence': answer_data['confidence'],
                'supporting_info': answer_data.get('supporting_info', []),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'context_length': len(context) if context else 0
                }
            }
            
        except (LLMConnectionError, LLMResponseError) as e:
            raise Exception(f"Question answering failed: {str(e)}")

    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create the prompt for the QnA task."""
        if context:
            return f"""
            Please answer the following question based on the provided context.
            If the answer cannot be fully determined from the context, indicate this clearly.

            Context:
            {context}

            Question: {question}

            Please provide your response in the following JSON format:
            {{
                "answer": "your detailed answer",
                "confidence": "high/medium/low",
                "supporting_info": ["relevant pieces of context that support the answer"]
            }}
            """
        else:
            return f"""
            Please answer the following question based on your knowledge:
            
            Question: {question}

            Please provide your response in the following JSON format:
            {{
                "answer": "your detailed answer",
                "confidence": "high/medium/low"
            }}
            """

    def _get_system_message(self) -> str:
        """Get the system message for QnA tasks."""
        return """You are a precise question-answering assistant. 
        Always provide accurate, well-reasoned answers with appropriate confidence levels. 
        If an answer cannot be determined with certainty, clearly state this and explain why."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format."""
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing if response is not valid JSON
            return {
                'answer': response,
                'confidence': 'low',
                'supporting_info': []
            }