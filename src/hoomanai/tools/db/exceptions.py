class DatabaseError(Exception):
    """Base class for database-related exceptions."""
    pass

class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass

class QueryError(DatabaseError):
    """Raised when query execution fails."""
    pass

class SchemaError(DatabaseError):
    """Raised when schema retrieval or validation fails."""
    pass

class ConfigurationError(DatabaseError):
    """Raised when database configuration is invalid."""
    pass