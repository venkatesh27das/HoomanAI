from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
from hoomanai.core.types import MemoryItem, MemoryType, MemoryPriority
from .storage import MemoryStorage
from .redis_storage import RedisStorage

class ShortTermMemory:
    """Manages short-term memory with fast access and automatic cleanup."""
    
    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        capacity: int = 100,
        default_ttl: timedelta = timedelta(hours=1)
    ):
        self.storage = storage or RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._initialize_cache()

    def _initialize_cache(self):
        """Load short-term memories into cache."""
        memories = self.storage.get_all(MemoryType.SHORT_TERM)
        for memory in memories:
            if not self._is_expired(memory):
                self._cache[memory.id] = memory
        self._enforce_capacity()

    def add(
        self,
        content: Any,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """Add a new item to short-term memory."""
        memory_item = MemoryItem(
            id=UUID(),
            content=content,
            created_at=datetime.now(),
            memory_type=MemoryType.SHORT_TERM,
            priority=priority,
            metadata=metadata or {},
            last_accessed=datetime.now(),
            access_count=0,
            ttl=datetime.now() + (ttl or self.default_ttl)
        )
        
        self._cache[memory_item.id] = memory_item
        self.storage.store(memory_item)
        self._enforce_capacity()
        
        return memory_item

    def get(self, memory_id: UUID) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        memory_item = self._cache.get(memory_id)
        
        if memory_item and not self._is_expired(memory_item):
            memory_item.last_accessed = datetime.now()
            memory_item.access_count += 1
            self.storage.store(memory_item)
            return memory_item
            
        if memory_item:
            self.remove(memory_id)
            
        return None

    def get_all(self) -> List[MemoryItem]:
        """Get all valid short-term memories."""
        current_time = datetime.now()
        valid_memories = []
        
        for memory_id, memory in list(self._cache.items()):
            if self._is_expired(memory):
                self.remove(memory_id)
            else:
                valid_memories.append(memory)
                
        return sorted(
            valid_memories,
            key=lambda x: (x.priority.value, x.last_accessed),
            reverse=True
        )

    def remove(self, memory_id: UUID) -> bool:
        """Remove a memory item."""
        if memory_id in self._cache:
            del self._cache[memory_id]
            self.storage.delete(memory_id)
            return True
        return False

    def clear(self) -> None:
        """Clear all short-term memories."""
        self._cache.clear()
        self.storage.clear(MemoryType.SHORT_TERM)

    def _is_expired(self, memory_item: MemoryItem) -> bool:
        """Check if a memory item has expired."""
        return memory_item.ttl and datetime.now() > memory_item.ttl

    def _enforce_capacity(self):
        """Ensure memory stays within capacity limits."""
        if len(self._cache) <= self.capacity:
            return
            
        # Sort by priority and last accessed time
        memories = list(self._cache.values())
        memories.sort(
            key=lambda x: (x.priority.value, x.last_accessed),
            reverse=True
        )
        
        # Remove excess memories
        for memory in memories[self.capacity:]:
            self.remove(memory.id)