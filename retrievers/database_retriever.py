from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional

class DatabaseRetriever:
    def __init__(self, db_url: str):
        self.engine: Engine = create_engine(db_url)

    def retrieve(self, sql_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return the results as a list of dictionaries.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql_query), parameters or {})
            rows = result.fetchall()
            keys = result.keys()
            return [dict(zip(keys, row)) for row in rows]
