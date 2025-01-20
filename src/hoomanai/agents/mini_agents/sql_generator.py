from typing import Any, Dict, Optional, List
from datetime import datetime

from hoomanai.core.mini_agent import MiniAgent
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.tools.llm.exceptions import LLMConnectionError, LLMResponseError
from hoomanai.core.types import Task

class SQLGeneratorAgent(MiniAgent):
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(config=llm_config)
        
    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute the SQL generation task.
        
        Args:
            task: Task object containing the natural language query and schema information
            
        Returns:
            Dictionary containing the generated SQL and explanation
            
        Raises:
            ValueError: If query or schema information is missing
            LLMConnectionError: If there's an error connecting to the LLM service
            LLMResponseError: If there's an error in the model's response
        """
        try:
            # Validate input
            if not task.context or 'query' not in task.context:
                raise ValueError("No natural language query provided")
                
            query = task.context['query']
            schema = task.context.get('schema', {})
            dialect = task.context.get('dialect', 'postgresql')
            
            # Create SQL generation prompt
            prompt = self._create_sql_prompt(query, schema, dialect)
            
            # Get SQL from LLM
            response = self.llm_client.complete(
                prompt=prompt,
                system_message=self._get_system_message(dialect)
            )
            
            # Parse and validate the response
            sql_data = self._parse_response(response.content)
            
            return {
                'sql_query': sql_data['sql'],
                'explanation': sql_data['explanation'],
                'tables_used': sql_data.get('tables_used', []),
                'metadata': {
                    'dialect': dialect,
                    'timestamp': datetime.now().isoformat(),
                    'complexity_score': self._calculate_complexity_score(sql_data['sql'])
                }
            }
            
        except (LLMConnectionError, LLMResponseError) as e:
            raise Exception(f"SQL generation failed: {str(e)}")

    def _create_sql_prompt(self, query: str, schema: Dict[str, Any], dialect: str) -> str:
        """Create the prompt for SQL generation."""
        schema_str = self._format_schema(schema)
        
        return f"""
        Please convert the following natural language query into {dialect} SQL.
        
        Database Schema:
        {schema_str}
        
        Natural Language Query:
        {query}
        
        Please provide your response in the following JSON format:
        {{
            "sql": "your SQL query",
            "explanation": "explanation of how the query works",
            "tables_used": ["list of tables used in the query"]
        }}
        
        Ensure the SQL query is optimized and follows best practices for {dialect}.
        """

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format the schema information into a readable string."""
        if not schema:
            return "No schema provided"
            
        formatted_schema = []
        for table, info in schema.items():
            columns = info.get('columns', {})
            column_str = '\n    '.join(f"{col}: {dtype}" for col, dtype in columns.items())
            formatted_schema.append(f"Table: {table}\n    {column_str}")
            
        return '\n\n'.join(formatted_schema)

    def _get_system_message(self, dialect: str) -> str:
        """Get the system message for SQL generation."""
        return f"""You are an expert {dialect} SQL generator. 
        Generate optimal, secure, and efficient SQL queries that follow best practices. 
        Always explain your queries and highlight any assumptions made."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format."""
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing if response is not valid JSON
            return {
                'sql': response,
                'explanation': 'No structured explanation available',
                'tables_used': []
            }

    def _calculate_complexity_score(self, sql: str) -> int:
        """Calculate a simple complexity score for the SQL query."""
        complexity = 0
        complexity_indicators = {
            'JOIN': 2,
            'WHERE': 1,
            'GROUP BY': 2,
            'HAVING': 3,
            'ORDER BY': 1,
            'UNION': 3,
            'INTERSECT': 3,
            'EXCEPT': 3,
            'WITH': 3,
            'WINDOW': 4
        }
        
        for indicator, score in complexity_indicators.items():
            if indicator in sql.upper():
                complexity += score
                
        return complexity