from typing import List, Dict, Any, Optional
import sqlalchemy
from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from .config import DatabaseConfig, DatabaseDialect
from .exceptions import ConnectionError, QueryError, SchemaError

class DatabaseConnector:
    """
    A unified database connector that handles connections to different SQL databases
    and provides schema information and query execution capabilities.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database connector with configuration.
        
        Args:
            config: DatabaseConfig object containing connection details
        """
        self.config = config
        self.engine = self._create_engine()
        self.dialect = config.dialect
        self._metadata = MetaData()
        self.logger = logging.getLogger(__name__)

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        """Create SQLAlchemy engine from config."""
        try:
            if self.config.connection_string:
                return create_engine(self.config.connection_string)
                
            # Build connection string from components
            conn_string = f"{self.config.dialect.value}://"
            
            if self.config.username and self.config.password:
                conn_string += f"{self.config.username}:{self.config.password}@"
                
            if self.config.host:
                conn_string += self.config.host
                
            if self.config.port:
                conn_string += f":{self.config.port}"
                
            conn_string += f"/{self.config.database}"
            
            return create_engine(conn_string)
            
        except Exception as e:
            raise ConnectionError(f"Failed to create database engine: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """
        Get database schema information including tables, columns, and relationships.
        
        Returns:
            Dictionary containing schema information
        """
        try:
            inspector = inspect(self.engine)
            schema = {}
            
            # Get all tables
            for table_name in inspector.get_table_names():
                columns = []
                primary_keys = []
                foreign_keys = []
                
                # Get column information
                for column in inspector.get_columns(table_name):
                    columns.append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column.get("nullable", True)
                    })
                    
                # Get primary key information
                for pk in inspector.get_pk_constraint(table_name).get("constrained_columns", []):
                    primary_keys.append(pk)
                    
                # Get foreign key information
                for fk in inspector.get_foreign_keys(table_name):
                    foreign_keys.append({
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                        "constrained_columns": fk["constrained_columns"]
                    })
                
                schema[table_name] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                }
            
            return schema
            
        except SQLAlchemyError as e:
            raise SchemaError(f"Failed to get database schema: {str(e)}")

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List of dictionaries containing query results
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                return [dict(row) for row in result]
                
        except SQLAlchemyError as e:
            raise QueryError(f"Query execution failed: {str(e)}")

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()

    @classmethod
    def from_connection_string(cls, connection_string: str, dialect: DatabaseDialect) -> 'DatabaseConnector':
        """Create connector instance from connection string."""
        config = DatabaseConfig(
            connection_string=connection_string,
            dialect=dialect
        )
        return cls(config)

    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table."""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)