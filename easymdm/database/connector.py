"""
Database connector for EasyMDM Advanced.
Supports multiple database types: SQLite, PostgreSQL, SQL Server, DuckDB, and CSV files.
"""

import os
import pandas as pd
import sqlite3
import duckdb
from typing import Optional, Dict, Any, Union, List
from abc import ABC, abstractmethod
from pathlib import Path
import logging

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

from ..core.config import DatabaseConfig

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
        
    @abstractmethod
    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from database."""
        pass
        
    @abstractmethod
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Save data to database."""
        pass
        
    @abstractmethod
    def test_connection(self) -> bool:
        """Test database connection."""
        pass


class CSVConnector(BaseConnector):
    """Connector for CSV files."""
    
    def connect(self) -> None:
        """CSV files don't require connection."""
        logger.info(f"CSV connector initialized for file: {self.config.file_path}")
        
    def disconnect(self) -> None:
        """CSV files don't require disconnection."""
        pass
        
    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            if not self.config.file_path:
                raise ValueError("File path not specified in configuration")
                
            if not os.path.exists(self.config.file_path):
                raise FileNotFoundError(f"CSV file not found: {self.config.file_path}")
                
            # Load CSV with options
            options = self.config.options or {}
            df = pd.read_csv(self.config.file_path, **options)
            
            # Reset index and add record_id
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'record_id'
            
            logger.info(f"Loaded {len(df)} records from CSV file: {self.config.file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV file {self.config.file_path}: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Save data to CSV file."""
        try:
            output_path = kwargs.get('output_path', self.config.file_path)
            options = kwargs.get('options', {})
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            df.to_csv(output_path, index=False, **options)
            logger.info(f"Saved {len(df)} records to CSV file: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV file {output_path}: {e}")
            raise
            
    def test_connection(self) -> bool:
        """Test CSV file accessibility."""
        try:
            if not self.config.file_path:
                return False
            return os.path.exists(self.config.file_path)
        except Exception:
            return False


class SQLiteConnector(BaseConnector):
    """Connector for SQLite databases."""
    
    def connect(self) -> None:
        """Establish SQLite connection."""
        try:
            db_path = self.config.file_path or self.config.database
            if not db_path:
                raise ValueError("Database path not specified")
                
            self.connection = sqlite3.connect(db_path)
            logger.info(f"Connected to SQLite database: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQLite database: {e}")
            raise
            
    def disconnect(self) -> None:
        """Close SQLite connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from SQLite database")
            
    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from SQLite database."""
        try:
            if not self.connection:
                self.connect()
                
            if query:
                df = pd.read_sql_query(query, self.connection)
            else:
                # Default query for table
                schema_table = f"{self.config.schema}.{self.config.table}" if self.config.schema else self.config.table
                query = f"SELECT * FROM {schema_table}"
                df = pd.read_sql_query(query, self.connection)
                
            # Reset index and add record_id
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'record_id'
            
            logger.info(f"Loaded {len(df)} records from SQLite database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from SQLite database: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Save data to SQLite database."""
        try:
            if not self.connection:
                self.connect()
                
            if_exists = kwargs.get('if_exists', 'replace')
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            
            logger.info(f"Saved {len(df)} records to SQLite table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to save data to SQLite table {table_name}: {e}")
            raise
            
    def test_connection(self) -> bool:
        """Test SQLite connection."""
        try:
            self.connect()
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception:
            return False
        finally:
            self.disconnect()


class PostgreSQLConnector(BaseConnector):
    """Connector for PostgreSQL databases."""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL connections")
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for PostgreSQL connections")
            
    def connect(self) -> None:
        """Establish PostgreSQL connection."""
        try:
            if self.config.connection_string:
                connection_string = self.config.connection_string
            else:
                connection_string = (
                    f"postgresql://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port or 5432}/{self.config.database}"
                )
                
            self.connection = create_engine(connection_string)
            logger.info(f"Connected to PostgreSQL database: {self.config.host}")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL database: {e}")
            raise
            
    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self.connection:
            self.connection.dispose()
            self.connection = None
            logger.info("Disconnected from PostgreSQL database")
            
    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from PostgreSQL database."""
        try:
            if not self.connection:
                self.connect()
                
            if query:
                df = pd.read_sql_query(query, self.connection)
            else:
                # Default query for table
                schema_table = f'"{self.config.schema}"."{self.config.table}"' if self.config.schema else f'"{self.config.table}"'
                query = f"SELECT * FROM {schema_table}"
                df = pd.read_sql_query(query, self.connection)
                
            # Reset index and add record_id
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'record_id'
            
            logger.info(f"Loaded {len(df)} records from PostgreSQL database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from PostgreSQL database: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Save data to PostgreSQL database."""
        try:
            if not self.connection:
                self.connect()
                
            if_exists = kwargs.get('if_exists', 'replace')
            schema = kwargs.get('schema', self.config.schema)
            
            df.to_sql(table_name, self.connection, schema=schema, 
                     if_exists=if_exists, index=False, method='multi')
            
            logger.info(f"Saved {len(df)} records to PostgreSQL table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to save data to PostgreSQL table {table_name}: {e}")
            raise
            
    def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            self.connect()
            with self.connection.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        finally:
            self.disconnect()


class SQLServerConnector(BaseConnector):
    """Connector for SQL Server databases."""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        if not PYODBC_AVAILABLE:
            raise ImportError("pyodbc is required for SQL Server connections")
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for SQL Server connections")
            
    def connect(self) -> None:
        """Establish SQL Server connection."""
        try:
            if self.config.connection_string:
                connection_string = self.config.connection_string
            else:
                driver = self.config.options.get('driver', 'ODBC Driver 17 for SQL Server')
                connection_string = (
                    f"mssql+pyodbc://{self.config.username}:{self.config.password}@"
                    f"{self.config.host}:{self.config.port or 1433}/{self.config.database}"
                    f"?driver={driver.replace(' ', '+')}"
                )
                
            self.connection = create_engine(connection_string)
            logger.info(f"Connected to SQL Server database: {self.config.host}")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server database: {e}")
            raise
            
    def disconnect(self) -> None:
        """Close SQL Server connection."""
        if self.connection:
            self.connection.dispose()
            self.connection = None
            logger.info("Disconnected from SQL Server database")
            
    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from SQL Server database."""
        try:
            if not self.connection:
                self.connect()
                
            if query:
                df = pd.read_sql_query(query, self.connection)
            else:
                # Default query for table
                schema_table = f"[{self.config.schema}].[{self.config.table}]" if self.config.schema else f"[{self.config.table}]"
                query = f"SELECT * FROM {schema_table}"
                df = pd.read_sql_query(query, self.connection)
                
            # Reset index and add record_id
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'record_id'
            
            logger.info(f"Loaded {len(df)} records from SQL Server database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from SQL Server database: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Save data to SQL Server database."""
        try:
            if not self.connection:
                self.connect()
                
            if_exists = kwargs.get('if_exists', 'replace')
            schema = kwargs.get('schema', self.config.schema)
            
            df.to_sql(table_name, self.connection, schema=schema, 
                     if_exists=if_exists, index=False, method='multi')
            
            logger.info(f"Saved {len(df)} records to SQL Server table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to save data to SQL Server table {table_name}: {e}")
            raise
            
    def test_connection(self) -> bool:
        """Test SQL Server connection."""
        try:
            self.connect()
            with self.connection.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        finally:
            self.disconnect()


class DuckDBConnector(BaseConnector):
    """Connector for DuckDB databases."""
    
    def connect(self) -> None:
        """Establish DuckDB connection."""
        try:
            db_path = self.config.file_path or self.config.database or ":memory:"
            self.connection = duckdb.connect(database=db_path, read_only=False)
            logger.info(f"Connected to DuckDB database: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB database: {e}")
            raise
            
    def disconnect(self) -> None:
        """Close DuckDB connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from DuckDB database")
            
    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load data from DuckDB database."""
        try:
            if not self.connection:
                self.connect()
                
            if query:
                df = self.connection.execute(query).df()
            else:
                # Default query for table
                table_name = self.config.table
                query = f"SELECT * FROM {table_name}"
                df = self.connection.execute(query).df()
                
            # Reset index and add record_id
            df.reset_index(drop=True, inplace=True)
            df.index.name = 'record_id'
            
            logger.info(f"Loaded {len(df)} records from DuckDB database")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from DuckDB database: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Save data to DuckDB database."""
        try:
            if not self.connection:
                self.connect()
                
            # Register DataFrame as a temporary table
            self.connection.register(f'temp_{table_name}', df)
            
            # Create or replace the table
            if_exists = kwargs.get('if_exists', 'replace')
            if if_exists == 'replace':
                query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_{table_name}"
            else:
                query = f"INSERT INTO {table_name} SELECT * FROM temp_{table_name}"
                
            self.connection.execute(query)
            
            logger.info(f"Saved {len(df)} records to DuckDB table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to save data to DuckDB table {table_name}: {e}")
            raise
            
    def test_connection(self) -> bool:
        """Test DuckDB connection."""
        try:
            self.connect()
            self.connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False
        finally:
            self.disconnect()


class DatabaseConnector:
    """Main database connector factory."""
    
    CONNECTOR_MAPPING = {
        'csv': CSVConnector,
        'sqlite': SQLiteConnector,
        'postgresql': PostgreSQLConnector,
        'sqlserver': SQLServerConnector,
        'duckdb': DuckDBConnector,
    }
    
    @classmethod
    def create_connector(cls, config: DatabaseConfig) -> BaseConnector:
        """Create appropriate connector based on database type."""
        db_type = config.type.lower()
        
        if db_type not in cls.CONNECTOR_MAPPING:
            raise ValueError(f"Unsupported database type: {db_type}")
            
        connector_class = cls.CONNECTOR_MAPPING[db_type]
        return connector_class(config)
    
    @classmethod
    def test_all_connections(cls) -> Dict[str, bool]:
        """Test all available database connections."""
        results = {}
        
        for db_type in cls.CONNECTOR_MAPPING:
            try:
                # Create a test configuration
                config = DatabaseConfig(type=db_type)
                connector = cls.create_connector(config)
                results[db_type] = True
                logger.info(f"{db_type} connector available")
            except Exception as e:
                results[db_type] = False
                logger.warning(f"{db_type} connector not available: {e}")
                
        return results
    
    @classmethod
    def get_available_connectors(cls) -> List[str]:
        """Get list of available database connectors."""
        available = []
        
        # CSV is always available
        available.append('csv')
        
        # SQLite is always available (built-in)
        available.append('sqlite')
        
        # DuckDB is always available if installed
        try:
            import duckdb
            available.append('duckdb')
        except ImportError:
            pass
            
        # PostgreSQL requires psycopg2 and SQLAlchemy
        if PSYCOPG2_AVAILABLE and SQLALCHEMY_AVAILABLE:
            available.append('postgresql')
            
        # SQL Server requires pyodbc and SQLAlchemy
        if PYODBC_AVAILABLE and SQLALCHEMY_AVAILABLE:
            available.append('sqlserver')
            
        return available


def create_connection_test(config: DatabaseConfig) -> bool:
    """Test a specific database connection configuration."""
    try:
        connector = DatabaseConnector.create_connector(config)
        return connector.test_connection()
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False