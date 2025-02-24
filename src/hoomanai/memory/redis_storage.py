from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
import json
import redis
from hoomanai.core.types import MemoryItem, MemoryType

class RedisStorage:
    def __init__(self, host='localhost', port=6379, db=0, prefix="memory:", **kwargs):
        # Separate prefix from Redis connection parameters
        self.key_prefix = prefix
        # Only pass Redis-specific parameters to Redis client
        redis_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['password', 'socket_timeout', 'socket_connect_timeout', 
                              'socket_keepalive', 'socket_keepalive_options', 
                              'connection_pool', 'unix_socket_path', 'encoding', 
                              'encoding_errors', 'charset', 'errors', 'decode_responses', 
                              'retry_on_timeout', 'ssl', 'ssl_keyfile', 'ssl_certfile', 
                              'ssl_cert_reqs', 'ssl_ca_certs', 'max_connections']}
        
        self.redis_client = redis.Redis(host=host, port=port, db=db, **redis_kwargs)
        
    def _serialize_memory_item(self, item: MemoryItem) -> str:
        """Serialize MemoryItem to JSON string."""
        return json.dumps({
            'id': str(item.id),
            'content': item.content,
            'created_at': item.created_at.isoformat(),
            'memory_type': item.memory_type.value,
            'priority': item.priority.value,
            'metadata': item.metadata,
            'last_accessed': item.last_accessed.isoformat(),
            'access_count': item.access_count,
            'embedding': item.embedding,
            'ttl': item.ttl.isoformat() if item.ttl else None
        })
        
    def _deserialize_memory_item(self, data: str) -> MemoryItem:
        """Deserialize JSON string to MemoryItem."""
        json_data = json.loads(data)
        return MemoryItem(
            id=UUID(json_data['id']),
            content=json_data['content'],
            created_at=datetime.fromisoformat(json_data['created_at']),
            memory_type=MemoryType(json_data['memory_type']),
            priority=MemoryPriority(json_data['priority']),
            metadata=json_data['metadata'],
            last_accessed=datetime.fromisoformat(json_data['last_accessed']),
            access_count=json_data['access_count'],
            embedding=json_data['embedding'],
            ttl=datetime.fromisoformat(json_data['ttl']) if json_data['ttl'] else None
        )
        
    def store(self, item: MemoryItem) -> None:
        """Store a memory item in Redis."""
        key = f"{self.key_prefix}{str(item.id)}"
        serialized_data = self._serialize_memory_item(item)
        
        # Store the memory item
        self.redis_client.set(key, serialized_data)
        
        # Set TTL if specified
        if item.ttl:
            ttl_seconds = (item.ttl - datetime.now()).total_seconds()
            if ttl_seconds > 0:
                self.redis_client.expire(key, int(ttl_seconds))
                
        # Add to type-specific set
        type_key = f"{self.key_prefix}type:{item.memory_type.value}"
        self.redis_client.sadd(type_key, str(item.id))
        
    def get(self, memory_id: UUID) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        key = f"{self.key_prefix}{str(memory_id)}"
        data = self.redis_client.get(key)
        return self._deserialize_memory_item(data.decode()) if data else None
        
    def get_all(self, memory_type: Optional[MemoryType] = None) -> List[MemoryItem]:
        """Retrieve all memory items, optionally filtered by type."""
        if memory_type:
            type_key = f"{self.key_prefix}type:{memory_type.value}"
            memory_ids = self.redis_client.smembers(type_key)
        else:
            pattern = f"{self.key_prefix}*"
            memory_ids = [key.replace(self.key_prefix.encode(), b'') 
                         for key in self.redis_client.keys(pattern)
                         if not key.decode().startswith(f"{self.key_prefix}type:")]
            
        memories = []
        for mid in memory_ids:
            try:
                memory_id = str(mid.decode())
                memory = self.get(UUID(memory_id))
                if memory:
                    memories.append(memory)
            except (ValueError, AttributeError):
                continue
                
        return memories
        
    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory item."""
        key = f"{self.key_prefix}{str(memory_id)}"
        memory = self.get(memory_id)
        
        if memory:
            # Remove from type-specific set
            type_key = f"{self.key_prefix}type:{memory.memory_type.value}"
            self.redis_client.srem(type_key, str(memory_id))
            
            # Delete the memory item
            return bool(self.redis_client.delete(key))
            
        return False
        
    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear all memories of a specific type or all memories."""
        if memory_type:
            type_key = f"{self.key_prefix}type:{memory_type.value}"
            memory_ids = self.redis_client.smembers(type_key)
            
            for mid in memory_ids:
                self.delete(UUID(mid.decode()))
            self.redis_client.delete(type_key)
        else:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)