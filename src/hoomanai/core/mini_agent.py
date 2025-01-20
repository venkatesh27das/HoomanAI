from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime

from hoomanai.core.types import Task
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError

class MiniAgent(ABC):
    """Base class for all mini agents in the system."""
    
    def __init__(self, name: str, description: str = "", llm_config: LLMConfig = None):
        self.name = name
        self.description = description
        self.llm_client = LLMClient(config=llm_config) if llm_config else None
        self.capabilities = self._init_capabilities()

    @abstractmethod
    def execute(self, task: Task) -> Dict[str, Any]:
        """Execute the agent's primary task.
        
        Args:
            task (Task): Task object containing execution context and requirements
            
        Returns:
            Dict[str, Any]: Results of task execution with status
        """
        pass

    def _init_capabilities(self) -> Dict[str, Any]:
        """Initialize agent capabilities."""
        return {
            'name': self.name,
            'description': self.description,
            'supported_tasks': self._get_supported_tasks(),
            'input_requirements': self._get_input_requirements(),
            'output_format': self._get_output_format()
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities."""
        return self.capabilities

    def _get_supported_tasks(self) -> List[str]:
        """Define tasks supported by the agent."""
        return []

    def _get_input_requirements(self) -> Dict[str, Any]:
        """Define required input format."""
        return {}

    def _get_output_format(self) -> Dict[str, Any]:
        """Define expected output format."""
        return {}

    def handle_error(self, task: Task, error: Exception) -> Dict[str, Any]:
        """Handle task execution errors.
        
        Args:
            task (Task): Failed task
            error (Exception): Error that occurred
            
        Returns:
            Dict[str, Any]: Error handling results
        """
        return {
            'status': 'error',
            'error': str(error),
            'task_id': str(task.id)
        }

    def validate_input(self, task: Task) -> bool:
        """Validate task input against requirements."""
        required_inputs = self._get_input_requirements()
        for key, value_type in required_inputs.items():
            if key not in task.input_context:
                return False
            if not isinstance(task.input_context[key], eval(value_type)):
                return False
        return True