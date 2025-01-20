from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import UUID

from .types import Task, TaskStatus

class MiniAgent(ABC):
    """
    Abstract base class for mini agents that perform specific tasks.
    Mini agents are the basic building blocks that handle single, focused tasks
    like summarization, QA, or SQL generation.
    """
    
    def __init__(self):
        """Initialize the mini agent with any required configurations."""
        pass
        
    @abstractmethod
    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the specific task assigned to this mini agent.
        
        Args:
            task: Task object containing the task details, context, and execution parameters
                 
        Returns:
            Dictionary containing the task results and any relevant metadata
            
        Raises:
            NotImplementedError: If the child class doesn't implement this method
            ValueError: If task parameters are invalid
        """
        raise NotImplementedError("Mini agents must implement execute method")
    
    def validate_task(self, task: Task) -> bool:
        """
        Validate that the task contains all required parameters.
        
        Args:
            task: Task object to validate
            
        Returns:
            bool: True if task is valid, False otherwise
        """
        if not hasattr(task, 'id') or not isinstance(task.id, UUID):
            return False
        if not hasattr(task, 'status') or not isinstance(task.status, TaskStatus):
            return False
        if not hasattr(task, 'description') or not task.description:
            return False
        return True