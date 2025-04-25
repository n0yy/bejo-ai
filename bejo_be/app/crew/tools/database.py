from crewai.tools import BaseTool
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import Field, PrivateAttr
from typing import Any, Optional, Dict
import json
import time
import logging
import hashlib
from cachetools import TTLCache
from dotenv import load_dotenv

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SQLTool(BaseTool):
    name: str = "SQL NL2SQL Tool"
    description: str = "Use this tool to execute SQL queries on the database."
    db_uri: str = Field(description="The URI of the database to connect to.")
    sample_limit: int = Field(
        default=3, description="Maximum number of sample data to fetch per column"
    )
    _engine: Any = PrivateAttr()
    _tables: list[str] = PrivateAttr(default_factory=list)
    _columns: dict = PrivateAttr(default_factory=dict)
    _sample_data: dict = PrivateAttr(default_factory=dict)
    _cache: TTLCache = PrivateAttr()
    _schema_hash: str = PrivateAttr(default="")
    _db_name: str = PrivateAttr(default="")

    def __init__(
        self,
        db_uri: str,
        sample_limit: int = 3,
        cache_ttl: int = 3600,
        cache_size: int = 100,
    ):
        """
        Initialize the SQLTool instance.

        Args:
            db_uri (str): The URI of the database to connect to.
            sample_limit (int, optional): Maximum number of sample data to fetch per column. Defaults to 20.
            cache_ttl (int, optional): Time to live for the cache in seconds. Defaults to 3600.
            cache_size (int, optional): Maximum number of items to store in the cache. Defaults to 100.
        """
        super().__init__(db_uri=db_uri, sample_limit=sample_limit)
        self._engine = create_engine(db_uri)
        self._cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self.cache_function = self._ttl_cache_func

        # Extract database name from URI
        self._db_name = db_uri.split("/")[-1].split("?")[0]

        # Metadata will be loaded lazily
        self._tables = []
        self._columns = {}
        self._sample_data = {}

    def _load_metadata_if_needed(self):
        """Lazy loading of metadata"""
        if not self._tables:
            try:
                self._fetch_metadata()
                self._update_schema_hash()
            except SQLAlchemyError as e:
                logger.error(f"Failed to load metadata: {str(e)}")
                raise

    def _update_schema_hash(self):
        """Generate a hash of current schema to detect changes"""
        schema_str = json.dumps(self._columns)
        self._schema_hash = hashlib.md5(schema_str.encode()).hexdigest()

    def _fetch_metadata(self):
        """Fetch tables and columns in one efficient query using information_schema"""
        try:
            with self._engine.connect() as connection:
                # Get tables and columns in a single query
                query = text(
                    """
                    SELECT TABLE_NAME, COLUMN_NAME 
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = :db_name
                    ORDER BY TABLE_NAME, ORDINAL_POSITION
                """
                )

                result = connection.execute(query, {"db_name": self._db_name})

                # Process results
                tables = set()
                columns = {}

                for row in result:
                    table_name, column_name = row

                    if table_name not in tables:
                        tables.add(table_name)
                        columns[table_name] = []

                    columns[table_name].append(column_name)

                self._tables = list(tables)
                self._columns = columns

        except SQLAlchemyError as e:
            logger.error(f"Error fetching metadata: {str(e)}")
            raise

    def get_sample_data(self, table_name=None):
        """Fetch sample data on demand, optionally for a specific table"""
        self._load_metadata_if_needed()

        tables_to_fetch = [table_name] if table_name else self._tables

        if table_name and table_name not in self._tables:
            raise ValueError(f"Table {table_name} not found in database")

        sample_data = {}

        try:
            with self._engine.connect() as connection:
                for table in tables_to_fetch:
                    if table not in self._sample_data:
                        row_sample = {}
                        for column in self._columns[table]:
                            # Use LIMIT directly without ORDER BY RAND() for efficiency
                            query = text(
                                f"SELECT DISTINCT {column} FROM {table} LIMIT :limit"
                            )
                            result = connection.execute(
                                query, {"limit": self.sample_limit}
                            )
                            row_sample[column] = [row[0] for row in result]

                        self._sample_data[table] = row_sample

                    sample_data[table] = self._sample_data[table]

            return sample_data

        except SQLAlchemyError as e:
            logger.error(f"Error fetching sample data: {str(e)}")
            raise

    def _get_context(self) -> str:
        """Get database schema and sample data context"""
        self._load_metadata_if_needed()

        # Get sample data for all tables if not already loaded
        if not self._sample_data:
            self.get_sample_data()

        return f"""
*Tables in the database:*
{', '.join(self._tables)}

*Columns in the database (Table Name -> Columns):*
{json.dumps(self._columns, indent=2)}

*Sample data in the database (Table Name -> Sample Data):*
```json
{json.dumps(self._sample_data, indent=4, default=str)}
```
        """

    def _ttl_cache_func(self, arguments: dict, result: Any) -> bool:
        """Improved caching function with query hashing"""
        query = arguments.get("query", "").strip()
        params = arguments.get("params", {})

        # Only cache SELECT type queries
        if query.upper().startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")):
            # Create a hash of the query and parameters for better caching
            param_str = json.dumps(params, sort_keys=True) if params else ""
            cache_key = hashlib.md5((query + param_str).encode()).hexdigest()

            self._cache[cache_key] = result
            return True

        return False

    def check_schema_changes(self):
        """Check if database schema has changed and invalidate cache if needed"""
        old_hash = self._schema_hash

        # Reload metadata
        self._fetch_metadata()
        self._update_schema_hash()

        # If hash changed, invalidate cache
        if old_hash != self._schema_hash:
            logger.info("Schema changed - invalidating cache")
            self._cache.clear()
            self._sample_data.clear()
            return True

        return False

    def _run(self, query: str, params: Optional[Dict] = None) -> str:
        """Execute SQL query with improved error handling and security"""
        # Check if the query type is allowed
        query_upper = query.strip().upper()
        if not query_upper.startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")):
            raise ValueError(
                "Only SELECT, SHOW, DESCRIBE, and EXPLAIN queries are allowed"
            )

        # Check for schema changes periodically
        self.check_schema_changes()

        # Check cache
        params = params or {}
        param_str = json.dumps(params, sort_keys=True) if params else ""
        cache_key = hashlib.md5((query + param_str).encode()).hexdigest()

        if cache_key in self._cache:
            logger.info("Cache hit for query")
            return self._cache[cache_key]

        # Execute query
        try:
            logger.info(f"Executing query: {query}")
            start_time = time.time()

            with self._engine.connect() as connection:
                result = connection.execute(text(query), params)
                rows = result.fetchall()

            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f} seconds")

            # Cache results if eligible
            if query_upper.startswith(("SELECT", "SHOW", "DESCRIBE", "EXPLAIN")):
                self._cache[cache_key] = rows

            return rows

        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise ValueError(f"Query execution error: {str(e)}")
