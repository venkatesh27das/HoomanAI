from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from hoomanai.core.types import Task, TaskStatus
from hoomanai.core.mini_agent import MiniAgent
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError

class CompoundAgent(ABC):
    """Base class for compound agents that orchestrate multiple mini agents."""
    
    def __init__(self, name: str, description: str = "", llm_config: LLMConfig = None):
        self.name = name
        self.description = description
        self.llm_client = LLMClient(config=llm_config) if llm_config else None
        self.mini_agents: Dict[str, MiniAgent] = {}
        self.capabilities = self._init_capabilities()
        self._workflow = self._define_workflow()

    @abstractmethod
    def execute(self, task: Task) -> Dict[str, Any]:
        """Execute the compound task using registered mini agents.
        
        Args:
            task (Task): Task object containing execution context
            
        Returns:
            Dict[str, Any]: Results of task execution
        """
        try:
            # Validate input
            if not self.validate_input(task):
                raise ValueError("Invalid task input")

            # Execute workflow
            results = {}
            for step in self._workflow:
                step_result = self._execute_workflow_step(step, task, results)
                results[step['name']] = step_result

            return self._consolidate_results(results)

        except Exception as e:
            return self.handle_error(task, e)

    def register_mini_agent(self, name: str, agent: MiniAgent):
        """Register a mini agent for use in the compound workflow."""
        self.mini_agents[name] = agent

    def _init_capabilities(self) -> Dict[str, Any]:
        """Initialize compound agent capabilities."""
        return {
            'name': self.name,
            'description': self.description,
            'supported_tasks': self._get_supported_tasks(),
            'input_requirements': self._get_input_requirements(),
            'output_format': self._get_output_format(),
            'workflow_steps': self._get_workflow_steps()
        }

    @abstractmethod
    def _define_workflow(self) -> List[Dict[str, Any]]:
        """Define the workflow steps and their dependencies.
        
        Returns:
            List[Dict[str, Any]]: List of workflow steps with their configurations
        """
        return []

    def _execute_workflow_step(self, step: Dict[str, Any], task: Task, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        agent = self.mini_agents.get(step['agent'])
        if not agent:
            raise ValueError(f"Agent {step['agent']} not found")

        # Prepare step input
        step_input = self._prepare_step_input(step, task, previous_results)
        
        # Create step task
        step_task = Task(
            id=UUID(int=0),  # Replace with proper UUID generation
            description=step['description'],
            agent_type="mini",
            agent_name=step['agent'],
            dependencies=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            input_context=step_input,
            expected_output=step.get('expected_output', {})
        )

        # Execute step
        return agent.execute(step_task)

    def _prepare_step_input(self, step: Dict[str, Any], task: Task, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a workflow step."""
        step_input = {}
        
        # Add original task context
        step_input.update(task.input_context)
        
        # Add results from previous steps
        for dep in step.get('dependencies', []):
            if dep in previous_results:
                step_input[f'prev_{dep}'] = previous_results[dep]
                
        return step_input

    def _consolidate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate results from all workflow steps."""
        return {
            'status': 'success',
            'results': results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'workflow_steps': list(results.keys())
            }
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return compound agent capabilities."""
        return self.capabilities

    def _get_supported_tasks(self) -> List[str]:
        """Define tasks supported by the compound agent."""
        return []

    def _get_input_requirements(self) -> Dict[str, Any]:
        """Define required input format."""
        return {}

    def _get_output_format(self) -> Dict[str, Any]:
        """Define expected output format."""
        return {}

    def _get_workflow_steps(self) -> List[Dict[str, Any]]:
        """Get workflow steps information."""
        return self._workflow

    def validate_input(self, task: Task) -> bool:
        """Validate task input against requirements."""
        required_inputs = self._get_input_requirements()
        for key, value_type in required_inputs.items():
            if key not in task.input_context:
                return False
            if not isinstance(task.input_context[key], eval(value_type)):
                return False
        return True

    def handle_error(self, task: Task, error: Exception) -> Dict[str, Any]:
        """Handle task execution errors."""
        return {
            'status': 'error',
            'error': str(error),
            'task_id': str(task.id)
        }