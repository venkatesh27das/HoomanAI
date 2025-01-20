from typing import Optional, Dict, Any, List
from hoomanai.core.compound_agent import CompoundAgent
from hoomanai.core.memory import MemoryManager
from hoomanai.agents.mini_agents.retriever_agent import RetrieverAgent
from hoomanai.agents.mini_agents.qna import QnAAgent
from hoomanai.tools.llm.config import LLMConfig

class RAG(CompoundAgent):
    """
    Retrieval Augmented Generation (RAG) compound agent that combines
    document retrieval with question answering capabilities.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        memory: Optional[MemoryManager] = None,
        similarity_threshold: float = 0.7,
        max_chunks: int = 5,
        name: str = "RAG"
    ):
        # Initialize component agents
        retriever = RetrieverAgent(
            llm_config=llm_config,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks
        )
        
        qna = QnAAgent(llm_config=llm_config)
        
        # Initialize compound agent with components
        super().__init__(
            agents=[retriever, qna],
            memory=memory,
            name=name
        )

    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the RAG pipeline.
        
        Args:
            task: Task object containing the question and documents
            
        Returns:
            Dictionary containing the final answer, relevant contexts,
            and execution metadata
        """
        try:
            # Prepare retrieval task context
            retrieval_context = {
                'query': task.context['question'],  # Use question as retrieval query
                'documents': task.context['documents']
            }
            
            # Update task context for retriever
            task.context.update(retrieval_context)
            
            # Execute compound agent pipeline
            results = super().execute(task)
            
            # Extract final answer and relevant contexts
            final_result = results['final_result']
            retrieval_result = results['intermediate_results'][0]
            
            return {
                'answer': final_result['answer'],
                'confidence': final_result['confidence'],
                'relevant_contexts': retrieval_result['retrieved_contexts'],
                'metadata': {
                    'retrieval_metadata': retrieval_result['metadata'],
                    'qna_metadata': final_result['metadata'],
                    'execution_sequence': results['metadata']['execution_sequence']
                }
            }
            
        except Exception as e:
            raise Exception(f"RAG execution failed: {str(e)}")

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        memory: Optional[MemoryManager] = None
    ) -> 'RAG':
        """
        Create a RAG instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing LLM and retrieval settings
            memory: Optional memory manager instance
            
        Returns:
            Configured RAG instance
        """
        llm_config = LLMConfig(**config.get('llm_config', {}))
        
        return cls(
            llm_config=llm_config,
            memory=memory,
            similarity_threshold=config.get('similarity_threshold', 0.7),
            max_chunks=config.get('max_chunks', 5),
            name=config.get('name', 'RAG')
        )