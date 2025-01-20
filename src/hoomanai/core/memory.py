from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import heapq
from .types import MemoryItem, MemoryType, MemoryPriority, ConversationContext
from hoomanai.memory.storage import MemoryStorage
from hoomanai.memory.redis_storage import RedisStorage
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.exceptions import LLMConnectionError

class MemoryManager:
    def __init__(
        self,
        llm_client: LLMClient,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        short_term_limit: int = 100,
        working_memory_limit: int = 10,
        default_ttl: timedelta = timedelta(hours=24),
        storage: Optional[MemoryStorage] = None,
    ):
        self.storage = storage or RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )
        self.llm_client = llm_client
        self.short_term_limit = short_term_limit
        self.working_memory_limit = working_memory_limit
        self.default_ttl = default_ttl
        
    def add_memory(
        self,
        content: Any,
        memory_type: MemoryType,
        priority: MemoryPriority,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[timedelta] = None
    ) -> MemoryItem:
        """Add a new memory item to the system."""
        memory_item = MemoryItem(
            id=uuid4(),
            content=content,
            created_at=datetime.now(),
            memory_type=memory_type,
            priority=priority,
            metadata=metadata or {},
            last_accessed=datetime.now(),
            access_count=0,
            embedding=None,
            ttl=datetime.now() + (ttl or self.default_ttl)
        )
        
        # Generate embedding for the content
        try:
            memory_item.embedding = self._generate_embedding(content)
        except LLMConnectionError:
            pass  # Continue even if embedding generation fails
            
        # Store the memory item
        self.storage.store(memory_item)
        
        # Manage memory limits
        self._enforce_memory_limits(memory_type)
        
        return memory_item

    def get_relevant_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        min_relevance_score: float = 0.7
    ) -> List[MemoryItem]:
        """Retrieve relevant memories based on a query."""
        query_embedding = self._generate_embedding(query)
        
        # Get candidate memories
        candidates = self.storage.get_all(memory_type) if memory_type else self.storage.get_all()
        
        # Filter expired memories
        candidates = [m for m in candidates if not self._is_expired(m)]
        
        # Score and rank memories
        scored_memories = []
        for memory in candidates:
            if memory.embedding:
                relevance_score = self._calculate_relevance(query_embedding, memory.embedding)
                if relevance_score >= min_relevance_score:
                    heapq.heappush(scored_memories, (-relevance_score, memory))
        
        # Return top relevant memories
        return [memory for _, memory in heapq.nheap(limit, scored_memories)]

    def update_memory(self, memory_id: UUID, updates: Dict[str, Any]) -> Optional[MemoryItem]:
        """Update an existing memory item."""
        memory_item = self.storage.get(memory_id)
        if not memory_item:
            return None
            
        for key, value in updates.items():
            if hasattr(memory_item, key):
                setattr(memory_item, key, value)
                
        memory_item.last_accessed = datetime.now()
        memory_item.access_count += 1
        
        self.storage.store(memory_item)
        return memory_item

    def forget_memory(self, memory_id: UUID) -> bool:
        """Remove a memory item from the system."""
        return self.storage.delete(memory_id)

    def consolidate_memories(self, conversation_context: ConversationContext) -> List[MemoryItem]:
        """Consolidate short-term memories into long-term memory."""
        memories_to_consolidate = [
            m for m in conversation_context.memory_items 
            if m.memory_type == MemoryType.SHORT_TERM
        ]
        
        if not memories_to_consolidate:
            return []
            
        try:
            # Use LLM to generate consolidated memory
            consolidated_content = self._consolidate_with_llm(memories_to_consolidate)
            
            # Create new long-term memory
            consolidated_memory = self.add_memory(
                content=consolidated_content,
                memory_type=MemoryType.LONG_TERM,
                priority=MemoryPriority.MEDIUM,
                metadata={
                    "source_memories": [str(m.id) for m in memories_to_consolidate],
                    "conversation_id": str(conversation_context.conversation_id)
                }
            )
            
            # Remove consolidated short-term memories
            for memory in memories_to_consolidate:
                self.forget_memory(memory.id)
                
            return [consolidated_memory]
            
        except LLMConnectionError:
            return []

    def _generate_embedding(self, content: Any) -> List[float]:
        """Generate embedding for content using LLM."""
        if isinstance(content, (dict, list)):
            content = str(content)
        
        response = self.llm_client.complete(
            prompt=f"Generate embedding for: {content}",
            system_message="You are an embedding generation assistant."
        )
        
        # Parse embedding from response
        # Note: This is a simplified version. In practice, you'd use a proper embedding model
        return [float(x) for x in response.content.split(",")]

    def _calculate_relevance(self, query_embedding: List[float], memory_embedding: List[float]) -> float:
        """Calculate relevance score between query and memory embeddings."""
        if len(query_embedding) != len(memory_embedding):
            return 0.0
            
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, memory_embedding))
        magnitude1 = sum(a * a for a in query_embedding) ** 0.5
        magnitude2 = sum(b * b for b in memory_embedding) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)

    def _is_expired(self, memory_item: MemoryItem) -> bool:
        """Check if a memory item has expired."""
        if not memory_item.ttl:
            return False
        return datetime.now() > memory_item.ttl

    def _enforce_memory_limits(self, memory_type: MemoryType):
        """Enforce memory limits by removing lowest priority items."""
        if memory_type == MemoryType.SHORT_TERM:
            limit = self.short_term_limit
        elif memory_type == MemoryType.WORKING:
            limit = self.working_memory_limit
        else:
            return

        memories = self.storage.get_all(memory_type)
        if len(memories) <= limit:
            return

        # Sort by priority and last accessed time
        memories.sort(
            key=lambda x: (x.priority.value, x.last_accessed),
            reverse=True
        )

        # Remove excess memories
        for memory in memories[limit:]:
            self.forget_memory(memory.id)

    def _consolidate_with_llm(self, memories: List[MemoryItem]) -> str:
        """Use LLM to consolidate multiple memories into a single coherent memory."""
        memories_content = "\n".join(
            f"Memory {i+1}: {m.content}" 
            for i, m in enumerate(memories)
        )
        
        prompt = f"""
        Please consolidate these related memories into a single coherent summary:
        
        {memories_content}
        
        Provide a concise but comprehensive summary that captures the key information.
        """
        
        response = self.llm_client.complete(
            prompt=prompt,
            system_message="You are a memory consolidation assistant."
        )
        
        return response.content