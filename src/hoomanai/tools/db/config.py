from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DatabaseDialect(Enum):
    POSTGRES = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"

@dataclass
class DatabaseConfig:
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    dialect: DatabaseDialect = DatabaseDialect.POSTGRES
    connection_string: Optional[str] = None