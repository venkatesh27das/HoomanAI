from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

from hoomanai.core.compound_agent import CompoundAgent
from hoomanai.core.memory import MemoryManager
from hoomanai.agents.mini_agents.sql_generator import SQLGeneratorAgent
from hoomanai.agents.mini_agents.qna import QnAAgent
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.core.types import Task, TaskStatus, AgentType
from hoomanai.tools.db.connector import DatabaseConnector  

class Query2Insight(CompoundAgent):
    """
    Query2Insight compound agent that converts natural language queries into SQL,
    executes them against a database, and provides human-readable insights from
    the results.
    """
    
    def __init__(
        self,
        db_connector: DatabaseConnector,
        llm_config: Optional[LLMConfig] = None,
        memory: Optional[MemoryManager] = None,
        name: str = "Query2Insight"
    ):
        # Initialize database connector
        self.db_connector = db_connector
        
        # Initialize component agents
        sql_generator = SQLGeneratorAgent(llm_config=llm_config)
        qna = QnAAgent(llm_config=llm_config)
        
        # Initialize compound agent with components
        super().__init__(
            agents=[sql_generator, qna],
            memory=memory,
            name=name
        )

    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the Query2Insight pipeline.
        
        Args:
            task: Task object containing the natural language query and database context
            
        Returns:
            Dictionary containing the insights, SQL query, raw data, and execution metadata
        """
        try:
            # Extract schema from database if not provided
            if 'schema' not in task.context:
                task.context['schema'] = self.db_connector.get_schema()

            # Execute compound agent pipeline
            results = super().execute(task)
            
            # Extract SQL generation and QnA results
            sql_result = results['intermediate_results'][0]
            insight_result = results['final_result']
            
            # Execute SQL query and get results
            query_results = self._execute_sql_query(sql_result['sql_query'])
            
            return {
                'insights': {
                    'summary': insight_result['answer'],
                    'confidence': insight_result['confidence'],
                    'supporting_points': insight_result.get('supporting_info', [])
                },
                'sql': {
                    'query': sql_result['sql_query'],
                    'explanation': sql_result['explanation'],
                    'tables_used': sql_result['tables_used'],
                    'complexity_score': sql_result['metadata']['complexity_score']
                },
                'data': {
                    'raw_results': query_results,
                    'row_count': len(query_results) if isinstance(query_results, list) else 0
                },
                'metadata': {
                    'execution_sequence': results['metadata']['execution_sequence'],
                    'timestamp': datetime.now().isoformat(),
                    'query_execution_time': results['metadata'].get('query_execution_time')
                }
            }
            
        except Exception as e:
            raise Exception(f"Query2Insight execution failed: {str(e)}")

    def _execute_sql_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query using database connector and return results."""
        try:
            return self.db_connector.execute_query(sql_query)
        except Exception as e:
            raise Exception(f"SQL query execution failed: {str(e)}")

    def _prepare_qna_context(
        self,
        query: str,
        sql_query: str,
        query_results: List[Dict[str, Any]],
        sql_explanation: str
    ) -> Dict[str, Any]:
        """Prepare context for the QnA agent to generate insights."""
        return {
            'question': f"""
                Based on the following SQL query and its results, please provide insights that answer this question: {query}
                
                SQL Query:
                {sql_query}
                
                SQL Explanation:
                {sql_explanation}
                
                Query Results:
                {self._format_results(query_results)}
            """,
            'context': sql_explanation
        }

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format query results into a readable string."""
        if not results:
            return "No results found"
            
        # Get columns from first result
        columns = list(results[0].keys())
        
        # Format header
        formatted = "| " + " | ".join(columns) + " |\n"
        formatted += "|" + "|".join(["-" * (len(col) + 2) for col in columns]) + "|\n"
        
        # Format rows
        for row in results[:10]:  # Limit to first 10 rows for readability
            formatted += "| " + " | ".join(str(row[col]) for col in columns) + " |\n"
            
        if len(results) > 10:
            formatted += "\n... and {len(results) - 10} more rows"
            
        return formatted

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        db_connector: DatabaseConnector,
        memory: Optional[MemoryManager] = None
    ) -> 'Query2Insight':
        """
        Create a Query2Insight instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing LLM and database settings
            db_connector: Database connector instance
            memory: Optional memory manager instance
            
        Returns:
            Configured Query2Insight instance
        """
        llm_config = LLMConfig(**config.get('llm_config', {}))
        
        return cls(
            db_connector=db_connector,
            llm_config=llm_config,
            memory=memory,
            name=config.get('name', 'Query2Insight')
        )

    def preprocess_query(self, query: str) -> Task:
        """
        Create a task object from a natural language query.
        """
        return Task(
            id=UUID(),
            description=f"Generate insights for: {query}",
            agent_type=AgentType.COMPOUND,
            agent_name=self.name,
            dependencies=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            context={
                'query': query,
                'schema': self.db_connector.get_schema(),
                'dialect': self.db_connector.dialect
            }
        )