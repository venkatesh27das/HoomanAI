from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Set
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
import json

from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.types import LLMResponse
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError

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
    input_context: Dict[str, Any]  # Added to store task-specific input
    expected_output: Dict[str, Any]  # Added to specify expected output format
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None

@dataclass
class ExecutionPlan:
    tasks: List[Task]
    context: UserContext
    metadata: Dict[str, Any]  # Added to store plan-level metadata

class MasterAgent:
    def __init__(self, conversation_memory=None, llm_config: Optional[LLMConfig] = None):
        self.conversation_memory = conversation_memory
        self.llm_client = LLMClient(config=llm_config)
        self.available_mini_agents = {}
        self.available_compound_agents = {}
        self.current_plan: Optional[ExecutionPlan] = None

    def register_mini_agent(self, name: str, agent_instance):
        """Register a mini agent with its capabilities description."""
        self.available_mini_agents[name] = {
            'instance': agent_instance,
            'capabilities': agent_instance.get_capabilities()
        }

    def register_compound_agent(self, name: str, agent_instance):
        """Register a compound agent with its capabilities description."""
        self.available_compound_agents[name] = {
            'instance': agent_instance,
            'capabilities': agent_instance.get_capabilities()
        }

    def create_execution_plan(self, user_context: UserContext) -> ExecutionPlan:
        """Create an LLM-driven execution plan based on the user's query and context."""
        try:
            # Get relevant context from conversation history
            historical_context = self._get_historical_context() if self.conversation_memory else None
            
            # First, analyze the query to understand requirements
            analysis_prompt = self._create_analysis_prompt(user_context, historical_context)
            analysis = self.llm_client.complete(
                prompt=analysis_prompt,
                system_message="""You are an expert system analyst. Analyze the query to understand:
                1. Core requirements
                2. Implicit requirements
                3. Required capabilities
                4. Potential challenges
                5. Success criteria"""
            )
            
            # Create planning prompt with analysis results
            plan_prompt = self._create_planning_prompt(user_context, historical_context, analysis.content)
            plan_response = self.llm_client.complete(
                prompt=plan_prompt,
                system_message="""You are an expert system architect. Create a detailed execution plan that:
                1. Breaks down the task into atomic steps
                2. Identifies the optimal agent for each step
                3. Defines clear dependencies
                4. Specifies required input/output for each step
                5. Includes error handling strategies"""
            )
            
            # Parse LLM response into structured tasks
            tasks = self._parse_plan_into_tasks(plan_response.content)
            
            # Validate the plan
            self._validate_plan(tasks)
            
            return ExecutionPlan(
                tasks=tasks,
                context=user_context,
                metadata={
                    'analysis': analysis.content,
                    'planning_response': plan_response.content,
                    'created_at': datetime.now()
                }
            )
            
        except (LLMConnectionError, LLMResponseError) as e:
            raise Exception(f"Failed to create execution plan: {str(e)}")

    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute the plan with improved orchestration and monitoring."""
        self.current_plan = plan
        results = {}
        error_handling_attempts = {}
        
        try:
            # Create execution graph
            execution_graph = self._create_execution_graph(plan.tasks)
            
            while not self._is_plan_complete():
                # Get tasks that can be executed in parallel
                ready_tasks = self._get_ready_tasks(execution_graph)
                
                # Execute ready tasks in parallel (if supported)
                for task in ready_tasks:
                    try:
                        # Update task status
                        task.status = TaskStatus.IN_PROGRESS
                        
                        # Get input context for task
                        task_input = self._prepare_task_input(task, results)
                        
                        # Execute task
                        result = self._execute_task(task, task_input)
                        
                        # Validate task output
                        if self._validate_task_output(task, result):
                            results[str(task.id)] = result
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now()
                            task.result = result
                        else:
                            raise ValueError("Task output validation failed")
                            
                    except Exception as e:
                        # Handle task failure
                        if str(task.id) not in error_handling_attempts:
                            error_handling_attempts[str(task.id)] = 0
                            
                        if error_handling_attempts[str(task.id)] < 3:
                            # Retry with error handling
                            retry_result = self._handle_task_error(task, e)
                            if retry_result:
                                results[str(task.id)] = retry_result
                                task.status = TaskStatus.COMPLETED
                                task.completed_at = datetime.now()
                                task.result = retry_result
                            else:
                                error_handling_attempts[str(task.id)] += 1
                                continue
                        else:
                            task.status = TaskStatus.FAILED
                            results[str(task.id)] = f"Failed: {str(e)}"
                
                # Update execution graph
                execution_graph = self._update_execution_graph(execution_graph, results)
                
            # Consolidate results
            final_results = self._consolidate_results(results)
            
            # Store execution history if memory available
            if self.conversation_memory:
                self._store_execution_history(plan, final_results)
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Plan execution failed: {str(e)}")

    def _create_analysis_prompt(self, user_context: UserContext, historical_context: Optional[Dict[str, Any]]) -> str:
        """Create a prompt for analyzing the user's query."""
        return f"""
        Analyze the following query and context to understand requirements and constraints:
        
        Query: {user_context.query}
        User Persona: {user_context.persona or 'None'}
        Additional Context: {json.dumps(user_context.additional_context) if user_context.additional_context else 'None'}
        Historical Context: {json.dumps(historical_context) if historical_context else 'None'}
        
        Please provide:
        1. Core requirements and objectives
        2. Implicit requirements or assumptions
        3. Required capabilities or skills
        4. Potential challenges or edge cases
        5. Success criteria and validation requirements
        
        Format your response as a structured JSON object.
        """


    def _create_planning_prompt(self, user_context: UserContext, historical_context: Optional[Dict[str, Any]], analysis: str) -> str:
        """Create an enhanced prompt for generating the execution plan."""
        # Get available agents and their capabilities
        mini_agents = {name: info['capabilities'] for name, info in self.available_mini_agents.items()}
        compound_agents = {name: info['capabilities'] for name, info in self.available_compound_agents.items()}
        
        # Prepare the JSON representations outside of the f-string
        user_context_json = json.dumps(user_context.additional_context) if user_context.additional_context else 'None'
        historical_context_json = json.dumps(historical_context) if historical_context else 'None'
        mini_agents_json = json.dumps(mini_agents, indent=2)
        compound_agents_json = json.dumps(compound_agents, indent=2)
        
        # Return the formatted string with pre-calculated values
        return f"""
        Based on the following analysis and available agents, create a detailed execution plan:
        
        Analysis: {analysis}
        
        Query: {user_context.query}
        User Context: {user_context_json}
        Historical Context: {historical_context_json}
        
        Available Agents and Capabilities:
        Mini Agents: {mini_agents_json}
        Compound Agents: {compound_agents_json}
        
        Create a plan that:
        1. Breaks down the task into atomic steps
        2. Assigns the most appropriate agent for each step
        3. Defines clear dependencies between steps
        4. Specifies required input/output for each step
        5. Includes error handling strategies
        
        Provide the plan in the following JSON format:
        {{
            "tasks": [
                {{
                    "description": "detailed task description",
                    "agent_type": "mini or compound",
                    "agent_name": "name of the agent",
                    "dependencies": ["list of task IDs this task depends on"],
                    "input_context": {{"required input parameters"}},
                    "expected_output": {{"expected output format"}},
                    "error_handling": {{"strategies for handling common errors"}}
                }}
            ],
            "metadata": {{
                "estimated_completion_time": "estimated time in seconds",
                "critical_path": ["ordered list of critical tasks"],
                "fallback_strategies": {{"alternative approaches if primary fails"}}
            }}
        }}
        """


    def _validate_plan(self, tasks: List[Task]) -> bool:
        """Validate the execution plan for completeness and correctness."""
        # Check for circular dependencies
        if self._has_circular_dependencies(tasks):
            raise ValueError("Execution plan contains circular dependencies")
            
        # Verify all required agents are available
        for task in tasks:
            if task.agent_type == AgentType.MINI and task.agent_name not in self.available_mini_agents:
                raise ValueError(f"Required mini agent {task.agent_name} not available")
            elif task.agent_type == AgentType.COMPOUND and task.agent_name not in self.available_compound_agents:
                raise ValueError(f"Required compound agent {task.agent_name} not available")
                
        # Validate input/output compatibility between dependent tasks
        for task in tasks:
            for dep_id in task.dependencies:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task:
                    if not self._are_io_compatible(dep_task.expected_output, task.input_context):
                        raise ValueError(f"Input/output mismatch between tasks {dep_task.id} and {task.id}")
                        
        return True

    def _prepare_task_input(self, task: Task, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a task based on its dependencies' results."""
        task_input = task.input_context.copy()
        
        # Add results from dependent tasks
        for dep_id in task.dependencies:
            if str(dep_id) in results:
                task_input[f'dep_{dep_id}'] = results[str(dep_id)]
                
        return task_input

    def _validate_task_output(self, task: Task, output: Any) -> bool:
        """Validate task output against expected format."""
        try:
            # Check if output matches expected schema
            for key, value_type in task.expected_output.items():
                if key not in output:
                    return False
                if not isinstance(output[key], eval(value_type)):
                    return False
            return True
        except Exception:
            return False

    def _handle_task_error(self, task: Task, error: Exception) -> Optional[Any]:
        """Handle task execution errors with retry logic and fallback strategies."""
        try:
            # Get agent instance
            agent = (self.available_mini_agents.get(task.agent_name, {}).get('instance') if task.agent_type == AgentType.MINI
                    else self.available_compound_agents.get(task.agent_name, {}).get('instance'))
            
            if hasattr(agent, 'handle_error'):
                return agent.handle_error(task, error)
            
            return None
        except Exception:
            return None

    def _store_execution_history(self, plan: ExecutionPlan, results: Dict[str, Any]):
        """Store execution history in conversation memory."""
        if self.conversation_memory:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'plan': {
                    'tasks': [{'id': str(t.id), 'description': t.description, 'status': t.status.value} for t in plan.tasks],
                    'metadata': plan.metadata
                },
                'results': results
            }
            self.conversation_memory.store('execution_history', history_entry)

    def _is_plan_complete(self) -> bool:
        """Check if all tasks in the current plan are completed or failed."""
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                  for task in self.current_plan.tasks)

    def _has_circular_dependencies(self, tasks: List[Task]) -> bool:
        """Check for circular dependencies in the task graph."""
        def has_cycle(task_id: UUID, visited: Set[UUID], stack: Set[UUID]) -> bool:
            visited.add(task_id)
            stack.add(task_id)
            
            task = next((t for t in tasks if t.id == task_id), None)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id, visited, stack):
                            return True
                    elif dep_id in stack:
                        return True
                        
            stack.remove(task_id)
            return False
            
        visited: Set[UUID] = set()
        stack: Set[UUID] = set()
        
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id, visited, stack):
                    return True
                    
        return False

    def _are_io_compatible(self, output_spec: Dict[str, Any], input_spec: Dict[str, Any]) -> bool:
        """Check if output and input specifications are compatible."""
        for key, value_type in input_spec.items():
            if key in output_spec and output_spec[key] != value_type:
                return False
        return True