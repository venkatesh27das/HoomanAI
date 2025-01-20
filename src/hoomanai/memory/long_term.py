from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import UUID
import numpy as np
from hoomanai.core.types import MemoryItem, MemoryType, MemoryPriority
from .storage import MemoryStorage
from hoomanai.tools.llm.client import LLMClient

class LongTermMemory:
    """Manages long-term memory with semantic search capabilities."""
    
    def __init__(
        self,
        storage: MemoryStorage,
        llm_client: LLMClient,
        similarity_threshold: float = 0.7,
        consolidation_threshold: int = 5
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        self.consolidation_threshold = consolidation_threshold

    def add(
        self,
        content: Any,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        check_duplicates: bool = True
    ) -> MemoryItem:
        """Add a new item to long-term memory."""
        # Generate embedding for the new content
        embedding = self._generate_embedding(content)
        
        if check_duplicates and embedding:
            # Check for similar existing memories
            similar_memories = self._find_similar_memories(embedding)
            if similar_memories:
                # If similar memories exist, consolidate them
                return self.consolidate([*similar_memories, MemoryItem(
                    id=UUID(),
                    content=content,
                    created_at=datetime.now(),
                    memory_type=MemoryType.LONG_TERM,
                    priority=priority,
                    metadata=metadata or {},
                    embedding=embedding
                )])

        memory_item = MemoryItem(
            id=UUID(),
            content=content,
            created_at=datetime.now(),
            memory_type=MemoryType.LONG_TERM,
            priority=priority,
            metadata=metadata or {},
            last_accessed=datetime.now(),
            access_count=0,
            embedding=embedding
        )
        
        self.storage.store(memory_item)
        return memory_item

    def get(self, memory_id: UUID) -> Optional[MemoryItem]:
        """Retrieve a specific memory item."""
        memory_item = self.storage.get(memory_id)
        
        if memory_item and memory_item.memory_type == MemoryType.LONG_TERM:
            memory_item.last_accessed = datetime.now()
            memory_item.access_count += 1
            self.storage.store(memory_item)
            return memory_item
            
        return None

    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """Search for relevant memories using semantic similarity."""
        query_embedding = self._generate_embedding(query)
        all_memories = self.storage.get_all(MemoryType.LONG_TERM)
        
        # Calculate similarities and filter results
        results = []
        min_sim = min_similarity or self.similarity_threshold
        
        for memory in all_memories:
            if not memory.embedding:
                continue
                
            # Check metadata filters if provided
            if filter_metadata and not self._matches_metadata_filter(memory, filter_metadata):
                continue
                
            similarity = self._calculate_similarity(query_embedding, memory.embedding)
            if similarity >= min_sim:
                results.append((memory, similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: (x[1], x[0].priority.value), reverse=True)
        return results[:limit]

    def consolidate(self, memories: List[MemoryItem]) -> Optional[MemoryItem]:
        """Consolidate multiple related memories into a single memory."""
        if not memories:
            return None
            
        # Prepare content for consolidation
        memory_contents = "\n".join(
            f"Memory {i+1} ({m.created_at.isoformat()}): {m.content}"
            for i, m in enumerate(memories)
        )
        
        # Use LLM to consolidate memories
        prompt = f"""
        Please consolidate these related memories into a single coherent summary:
        
        {memory_contents}
        
        Create a comprehensive summary that:
        1. Preserves key information from all memories
        2. Eliminates redundancy
        3. Maintains chronological order where relevant
        4. Highlights important patterns or connections
        
        Provide the consolidated memory in a clear, well-structured format.
        """
        
        try:
            response = self.llm_client.complete(
                prompt=prompt,
                system_message="You are a memory consolidation assistant."
            )
            
            # Merge metadata from all memories
            merged_metadata = self._merge_metadata(memories)
            merged_metadata.update({
                "source_memories": [str(m.id) for m in memories],
                "consolidated_at": datetime.now().isoformat(),
                "original_count": len(memories),
                "consolidation_version": merged_metadata.get("consolidation_version", 0) + 1
            })
            
            # Create new consolidated memory
            consolidated_memory = self.add(
                content=response.content,
                priority=max(m.priority for m in memories),
                metadata=merged_metadata,
                check_duplicates=False  # Avoid recursive consolidation
            )
            
            # Remove original memories
            for memory in memories:
                self.storage.delete(memory.id)
                
            return consolidated_memory
            
        except Exception as e:
            print(f"Error during memory consolidation: {e}")
            return None

    def forget(self, memory_id: UUID) -> bool:
        """Remove a memory item from long-term storage."""
        return self.storage.delete(memory_id)

    def update_metadata(self, memory_id: UUID, metadata_updates: Dict[str, Any]) -> Optional[MemoryItem]:
        """Update metadata for a specific memory item."""
        memory_item = self.get(memory_id)
        if not memory_item:
            return None
            
        memory_item.metadata.update(metadata_updates)
        self.storage.store(memory_item)
        return memory_item

    def _generate_embedding(self, content: Any) -> List[float]:
        """Generate embedding for content using LLM."""
        if isinstance(content, (dict, list)):
            content = str(content)
            
        try:
            response = self.llm_client.complete(
                prompt=f"Generate embedding for: {content}",
                system_message="You are an embedding generation assistant."
            )
            
            # Parse embedding from response
            # Note: In production, use a proper embedding model
            return [float(x) for x in response.content.split(",")]
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0
            
        try:
            # Convert to numpy arrays for efficient computation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def _find_similar_memories(self, embedding: List[float], threshold: Optional[float] = None) -> List[MemoryItem]:
        """Find existing memories similar to the given embedding."""
        if not embedding:
            return []
            
        threshold = threshold or self.similarity_threshold
        similar_memories = []
        
        all_memories = self.storage.get_all(MemoryType.LONG_TERM)
        for memory in all_memories:
            if memory.embedding and self._calculate_similarity(embedding, memory.embedding) >= threshold:
                similar_memories.append(memory)
                
        return similar_memories

    def _matches_metadata_filter(self, memory: MemoryItem, filter_metadata: Dict[str, Any]) -> bool:
        """Check if memory metadata matches the filter criteria."""
        for key, value in filter_metadata.items():
            if key not in memory.metadata:
                return False
            if isinstance(value, (list, set)):
                if memory.metadata[key] not in value:
                    return False
            elif memory.metadata[key] != value:
                return False
        return True

    def _merge_metadata(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Merge metadata from multiple memories."""
        merged = {}
        for memory in memories:
            for key, value in memory.metadata.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], list):
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)
                else:
                    merged[key] = [merged[key], value]
        return merged

    def analyze_patterns(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Analyze patterns and relationships between memories."""
        if not memories:
            return {}
            
        # Prepare memory contents for analysis
        memory_contents = "\n".join(
            f"Memory {i+1} ({m.created_at.isoformat()}): {m.content}"
            for i, m in enumerate(memories)
        )
        
        prompt = f"""
        Analyze the following memories and identify patterns, relationships, and insights:

        {memory_contents}

        Please provide analysis in JSON format with the following structure:
        {{
            "patterns": ["list of identified patterns"],
            "relationships": ["list of relationships between memories"],
            "key_concepts": ["list of important recurring concepts"],
            "timeline_insights": ["insights about temporal aspects"],
            "recommendations": ["suggestions for memory organization or consolidation"]
        }}
        """
        
        try:
            response = self.llm_client.complete(
                prompt=prompt,
                system_message="You are a memory analysis assistant."
            )
            
            return json.loads(response.content)
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {}