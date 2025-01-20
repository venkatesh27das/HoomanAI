from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
import json
import os
from hoomanai.core.types import MemoryItem, MemoryType
import sqlite3
import pickle

class MemoryStorage:
    """Handles persistent storage of memory items using SQLite."""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the SQLite database and create necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content BLOB,
                    created_at TIMESTAMP,
                    memory_type TEXT,
                    priority INTEGER,
                    metadata BLOB,
                    last_accessed TIMESTAMP,
                    access_count INTEGER,
                    embedding BLOB,
                    ttl TIMESTAMP
                )
            """)
            
            # Create index for faster querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type 
                ON memories(memory_type)
            """)
            
            conn.commit()

    def store(self, memory_item: MemoryItem) -> bool:
        """Store a memory item in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, created_at, memory_type, priority, metadata, 
                     last_accessed, access_count, embedding, ttl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(memory_item.id),
                    pickle.dumps(memory_item.content),
                    memory_item.created_at.isoformat(),
                    memory_item.memory_type.value,
                    memory_item.priority.value,
                    pickle.dumps(memory_item.metadata),
                    memory_item.last_accessed.isoformat() if memory_item.last_accessed else None,
                    memory_item.access_count,
                    pickle.dumps(memory_item.embedding) if memory_item.embedding else None,
                    memory_item.ttl.isoformat() if memory_item.ttl else None
                ))
                
                return True
                
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    def get(self, memory_id: UUID) -> Optional[MemoryItem]:
        """Retrieve a specific memory item by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM memories WHERE id = ?", (str(memory_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_memory_item(row)
                return None
                
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return None

    def get_all(self, memory_type: Optional[MemoryType] = None) -> List[MemoryItem]:
        """Retrieve all memory items, optionally filtered by type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if memory_type:
                    cursor.execute(
                        "SELECT * FROM memories WHERE memory_type = ?",
                        (memory_type.value,)
                    )
                else:
                    cursor.execute("SELECT * FROM memories")
                    
                return [self._row_to_memory_item(row) for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory item from storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE id = ?", (str(memory_id),))
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def clear(self, memory_type: Optional[MemoryType] = None) -> bool:
        """Clear all memories of a specific type or all memories if type not specified."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if memory_type:
                    cursor.execute(
                        "DELETE FROM memories WHERE memory_type = ?",
                        (memory_type.value,)
                    )
                else:
                    cursor.execute("DELETE FROM memories")
                    
                return True
                
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False

    def _row_to_memory_item(self, row) -> MemoryItem:
        """Convert a database row to a MemoryItem object."""
        return MemoryItem(
            id=UUID(row[0]),
            content=pickle.loads(row[1]),
            created_at=datetime.fromisoformat(row[2]),
            memory_type=MemoryType(row[3]),
            priority=int(row[4]),
            metadata=pickle.loads(row[5]),
            last_accessed=datetime.fromisoformat(row[6]) if row[6] else None,
            access_count=row[7],
            embedding=pickle.loads(row[8]) if row[8] else None,
            ttl=datetime.fromisoformat(row[9]) if row[9] else None
        )