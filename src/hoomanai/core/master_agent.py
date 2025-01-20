from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.types import LLMResponse
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError
import json

# First, let's define our types
class AgentType(Enum):
    MINI = "mini"
    COMPOUND = "compound"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class UserContext:
    query: str
    persona: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

@dataclass
class Task:
    id: UUID
    description: str
    agent_type: AgentType
    agent_name: str
    dependencies: List[UUID]
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None

@dataclass
class ExecutionPlan:
    tasks: List[Task]
    context: UserContext

class MasterAgent:
    def __init__(self, conversation_memory=None, llm_config: Optional[LLMConfig] = None):
        self.conversation_memory = conversation_memory
        self.llm_client = LLMClient(config=llm_config)
        self.available_mini_agents = {}  # Will be populated with registered mini agents
        self.available_compound_agents = {}  # Will be populated with registered compound agents
        self.current_plan: Optional[ExecutionPlan] = None

    def register_mini_agent(self, name: str, agent_instance):
        """Register a mini agent for use by the master agent."""
        self.available_mini_agents[name] = agent_instance

    def register_compound_agent(self, name: str, agent_instance):
        """Register a compound agent for use by the master agent."""
        self.available_compound_agents[name] = agent_instance

    def create_execution_plan(self, user_context: UserContext) -> ExecutionPlan:
        """
        Create an execution plan based on the user's query and context.
        This method uses LLM to analyze the query and create appropriate tasks.
        """
        try:
            # Get relevant context from conversation history
            historical_context = self._get_historical_context() if self.conversation_memory else None
            
            # Create and send the planning prompt
            plan_prompt = self._create_planning_prompt(user_context, historical_context)
            
            # Get completion from LLM
            plan_response = self.llm_client.complete(
                prompt=plan_prompt,
                system_message="You are a task planning assistant. Your role is to break down user queries into specific, actionable tasks that can be executed by different types of agents."
            )
            
            # Parse LLM response into structured tasks
            tasks = self._parse_plan_into_tasks(plan_response.content)
            
            return ExecutionPlan(tasks=tasks, context=user_context)
            
        except (LLMConnectionError, LLMResponseError) as e:
            raise Exception(f"Failed to create execution plan: {str(e)}")

    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute the created plan by running tasks in the correct order.
        """
        self.current_plan = plan
        results = {}
        
        # Create a map of task dependencies
        dependency_map = self._create_dependency_map(plan.tasks)
        
        # Execute tasks in order of dependencies
        while not self._is_plan_complete():
            ready_tasks = self._get_ready_tasks(dependency_map)
            
            for task in ready_tasks:
                try:
                    result = self._execute_task(task)
                    results[str(task.id)] = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.result = result
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    results[str(task.id)] = f"Failed: {str(e)}"

        return self._consolidate_results(results)

    def _get_historical_context(self) -> Dict[str, Any]:
        """Retrieve relevant context from conversation history."""
        if not self.conversation_memory:
            return {}
        return self.conversation_memory.get_relevant_context()

    def _create_planning_prompt(self, user_context: UserContext, historical_context: Optional[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM to generate an execution plan."""
        available_mini_agents = list(self.available_mini_agents.keys())
        available_compound_agents = list(self.available_compound_agents.keys())
        
        prompt = f"""
        Please analyze the following query and create a detailed execution plan:

        Query: {user_context.query}
        User Persona: {user_context.persona or 'None'}
        Additional Context: {json.dumps(user_context.additional_context) if user_context.additional_context else 'None'}
        Historical Context: {json.dumps(historical_context) if historical_context else 'None'}
        
        Available Agents:
        - Mini Agents: {', '.join(available_mini_agents)}
        - Compound Agents: {', '.join(available_compound_agents)}
        
        Please provide a response in the following JSON format:
        {{
            "tasks": [
                {{
                    "description": "detailed task description",
                    "agent_type": "mini or compound",
                    "agent_name": "name of the agent to execute this task",
                    "dependencies": ["list of task IDs this task depends on"],
                    "expected_output": "description of expected output format"
                }}
            ]
        }}
        
        Ensure that:
        1. Each task is atomic and clearly defined
        2. Dependencies are properly specified
        3. The selected agent is appropriate for the task
        4. The expected output format is clearly specified
        """
        
        return prompt

    def _parse_plan_into_tasks(self, plan_response: str) -> List[Task]:
        """Parse the LLM's response into structured Task objects."""
        try:
            # Attempt to parse the response as JSON
            plan_data = json.loads(plan_response)
            tasks = []
            
            for task_data in plan_data.get("tasks", []):
                task = Task(
                    id=uuid4(),
                    description=task_data.get("description", ""),
                    agent_type=AgentType(task_data.get("agent_type", "mini").lower()),
                    agent_name=task_data.get("agent_name", ""),
                    dependencies=[UUID(dep) for dep in task_data.get("dependencies", [])],
                    status=TaskStatus.PENDING,
                    created_at=datetime.now()
                )
                tasks.append(task)
                
            return tasks
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response into tasks: {str(e)}")


    def _create_dependency_map(self, tasks: List[Task]) -> Dict[UUID, List[UUID]]:
        """Create a map of task dependencies."""
        dependency_map = {}
        for task in tasks:
            dependency_map[task.id] = task.dependencies
        return dependency_map

    def _get_ready_tasks(self, dependency_map: Dict[UUID, List[UUID]]) -> List[Task]:
        """Get tasks that are ready to be executed (all dependencies completed)."""
        ready_tasks = []
        for task in self.current_plan.tasks:
            if task.status == TaskStatus.PENDING:
                dependencies = dependency_map[task.id]
                if all(self._is_task_completed(dep_id) for dep_id in dependencies):
                    ready_tasks.append(task)
        return ready_tasks

    def _is_task_completed(self, task_id: UUID) -> bool:
        """Check if a task is completed."""
        for task in self.current_plan.tasks:
            if task.id == task_id:
                return task.status == TaskStatus.COMPLETED
        return False

    def _is_plan_complete(self) -> bool:
        """Check if all tasks in the current plan are completed."""
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                  for task in self.current_plan.tasks)

    def _execute_task(self, task: Task) -> Any:
        """Execute a single task using the appropriate agent."""
        if task.agent_type == AgentType.MINI:
            agent = self.available_mini_agents.get(task.agent_name)
        else:
            agent = self.available_compound_agents.get(task.agent_name)
            
        if not agent:
            raise ValueError(f"Agent {task.agent_name} not found")
            
        return agent.execute(task)

    def _consolidate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate all task results into a final output."""
        # This method would implement logic to combine results from all tasks
        # into a coherent final output
        return {
            "final_result": results,
            "execution_summary": {
                "total_tasks": len(self.current_plan.tasks),
                "completed_tasks": len([t for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED]),
                "failed_tasks": len([t for t in self.current_plan.tasks if t.status == TaskStatus.FAILED])
            }
        }