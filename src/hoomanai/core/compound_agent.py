from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID

from .mini_agent import MiniAgent
from .types import Task, TaskStatus, AgentType
from .memory import MemoryManager

class CompoundAgent:
    """
    A compound agent that orchestrates multiple mini agents to accomplish
    more complex tasks. It manages the workflow between different mini agents
    and handles the passing of information between them.
    """
    
    def __init__(
        self,
        agents: List[MiniAgent],
        memory: Optional[Memory] = None,
        name: str = "CompoundAgent"
    ):
        """
        Initialize the compound agent with its component mini agents.
        
        Args:
            agents: List of mini agents that this compound agent will coordinate
            memory: Optional memory system for storing intermediate results
            name: Name identifier for the compound agent
        """
        self.agents = agents
        self.memory = memory
        self.name = name
        
    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute a complex task by coordinating multiple mini agents.
        
        Args:
            task: Task object containing the task details and execution context
                 
        Returns:
            Dictionary containing the final results and execution metadata
            
        Raises:
            ValueError: If task parameters are invalid
            Exception: If any mini agent execution fails
        """
        results = []
        metadata = {
            'agent_name': self.name,
            'num_agents': len(self.agents),
            'execution_sequence': [],
            'start_time': datetime.now().isoformat(),
        }
        
        try:
            current_context = task.context if hasattr(task, 'context') else {}
            
            # Execute each agent in sequence, passing results forward
            for idx, agent in enumerate(self.agents):
                # Create subtask with current context
                subtask = Task(
                    id=UUID(),
                    description=f"Subtask {idx + 1} of {task.description}",
                    agent_type=AgentType.MINI,
                    agent_name=agent.__class__.__name__,
                    dependencies=[],
                    status=TaskStatus.IN_PROGRESS,
                    created_at=datetime.now(),
                    context=current_context
                )
                
                # Execute the mini agent
                result = agent.execute(subtask)
                
                # Store result in memory if available
                if self.memory:
                    self.memory.store(f"{task.id}_{agent.__class__.__name__}", result)
                
                # Update context for next agent
                if isinstance(result, dict):
                    current_context.update(result)
                
                # Track execution
                results.append(result)
                metadata['execution_sequence'].append(agent.__class__.__name__)
                
            metadata['end_time'] = datetime.now().isoformat()
            
            return {
                'final_result': results[-1],
                'intermediate_results': results[:-1],
                'metadata': metadata
            }
            
        except Exception as e:
            metadata['end_time'] = datetime.now().isoformat()
            metadata['error'] = str(e)
            raise Exception(f"Compound agent execution failed: {str(e)}")
            
    def add_agent(self, agent: MiniAgent) -> None:
        """Add a new mini agent to the compound agent's workflow."""
        self.agents.append(agent)
        
    def remove_agent(self, index: int) -> None:
        """Remove a mini agent from the workflow by index."""
        if 0 <= index < len(self.agents):
            self.agents.pop(index)