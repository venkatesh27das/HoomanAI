from typing import Any, Dict, Optional, List
from datetime import datetime

from hoomanai.core.mini_agent import MiniAgent
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.core.types import Task

class RetrieverAgent(MiniAgent):
    def __init__(
        self, 
        llm_config: Optional[LLMConfig] = None,
        similarity_threshold: float = 0.7,
        max_chunks: int = 5
    ):
        self.llm_client = LLMClient(config=llm_config)
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks

    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the retrieval task to find relevant context.
        
        Args:
            task: Task object containing the query and document collection
            
        Returns:
            Dictionary containing retrieved contexts and metadata
        """
        try:
            # Validate input
            if not task.context or 'query' not in task.context:
                raise ValueError("No query provided")
            if 'documents' not in task.context:
                raise ValueError("No documents provided")

            query = task.context['query']
            documents = task.context['documents']

            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Process and retrieve relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(
                query_embedding,
                documents
            )

            return {
                'retrieved_contexts': relevant_chunks,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_chunks': len(relevant_chunks),
                    'total_documents': len(documents)
                }
            }

        except Exception as e:
            raise Exception(f"Retrieval failed: {str(e)}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using LLM."""
        response = self.llm_client.complete(
            prompt=f"Generate embedding for: {text}",
            system_message="You are an embedding generation assistant."
        )
        return [float(x) for x in response.content.split(",")]

    def _retrieve_relevant_chunks(
        self,
        query_embedding: List[float],
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Retrieve relevant document chunks based on similarity."""
        scored_chunks = []

        for doc in documents:
            # Process document into chunks (simplified)
            chunks = self._chunk_document(doc['content'])
            
            for chunk in chunks:
                chunk_embedding = self._generate_embedding(chunk)
                similarity = self._calculate_similarity(
                    query_embedding,
                    chunk_embedding
                )
                
                if similarity >= self.similarity_threshold:
                    scored_chunks.append((similarity, chunk))

        # Sort by similarity and take top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:self.max_chunks]]

    def _chunk_document(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Split document into chunks (simplified implementation)."""
        chunks = []
        words = content.split()
        current_chunk = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1
            if current_length > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        if len(embedding1) != len(embedding2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)