from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime
from uuid import UUID

# Memory-related enums
class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EXECUTION = "execution"

class MemoryPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

# Agent-related enums
class AgentType(Enum):
    MINI = "mini"
    COMPOUND = "compound"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# Memory-related types
@dataclass
class MemoryItem:
    id: UUID
    content: Any
    created_at: datetime
    memory_type: MemoryType
    priority: MemoryPriority
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    embedding: Optional[List[float]] = None
    ttl: Optional[datetime] = None

@dataclass
class AgentMemoryContext:
    short_term: List[MemoryItem]
    working: List[MemoryItem]
    relevant_long_term: List[MemoryItem]

# Agent-related types
@dataclass
class AgentAction:
    name: str
    description: str
    parameters: Dict[str, Any]
    required_capabilities: List[str]
    timeout: Optional[float] = None

@dataclass
class AgentCapability:
    name: str
    description: str
    required_tools: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    id: UUID
    status: str
    current_task: Optional[str]
    memory_context: AgentMemoryContext
    capabilities: List[AgentCapability]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_updates: List[MemoryItem] = field(default_factory=list)

# Task and Context types
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
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserContext:
    query: str
    persona: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

@dataclass
class ConversationContext:
    conversation_id: UUID
    start_time: datetime
    participants: List[str]
    metadata: Dict[str, Any]
    memory_items: List[MemoryItem] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    tasks: List[Task]
    context: UserContext
    metadata: Dict[str, Any] = field(default_factory=dict)

# Communication types
@dataclass
class AgentMessage:
    id: UUID
    sender: str
    receiver: str
    content: Any
    timestamp: datetime
    message_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResponse:
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)