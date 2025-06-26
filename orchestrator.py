from retrievers.vector_retriever import VectorRetriever
from retrievers.graph_retriever import GraphRetriever
from retrievers.database_retriever import DatabaseRetriever
from retrievers.keyword_retriever import KeywordRetriever
from typing import Dict, Any, List, Optional

class Orchestrator:
    def __init__(
        self,
        vector_cfg: Dict[str, Any],
        graph_cfg: Dict[str, Any],
        db_cfg: Dict[str, Any],
        keyword_cfg: Dict[str, Any],
    ):
        self.vector = VectorRetriever(**vector_cfg)
        self.graph = GraphRetriever(**graph_cfg)
        self.database = DatabaseRetriever(**db_cfg)
        self.keyword = KeywordRetriever(**keyword_cfg)

    def process_query(self, query: str, query_type: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Route the query to the appropriate retrievers based on query_type or simple heuristics.
        Returns a dictionary with results from each retriever.
        """
        results = {}
        # Simple rule-based routing; can be replaced with ML-based intent detection
        if query_type == "author":
            # Author lookup: use graph and database
            results["graph"] = self.graph.retrieve(self._build_author_cypher(query))
            results["database"] = self.database.retrieve(self._build_author_sql(query))
        elif query_type == "recent":
            # Recent papers: use database and keyword
            results["database"] = self.database.retrieve(self._build_recent_sql(query))
            results["keyword"] = self.keyword.retrieve(query, top_k=top_k)
        else:
            # Default: semantic, keyword, and graph
            results["vector"] = self.vector.retrieve(query, top_k=top_k)
            results["keyword"] = self.keyword.retrieve(query, top_k=top_k)
            # Optionally, graph and database for exploratory queries
        return results

    def _build_author_cypher(self, query: str) -> str:
        # Naive example: extract author name from query
        author = query.replace("author:", "").strip()
        return f"MATCH (a:Author)-[:AUTHORED]->(p:Paper) WHERE a.name CONTAINS '{author}' RETURN p, a"

    def _build_author_sql(self, query: str) -> str:
        author = query.replace("author:", "").strip()
        return f"SELECT * FROM papers p JOIN paper_authors pa ON p.id = pa.paper_id JOIN authors a ON pa.author_id = a.id WHERE a.name ILIKE '%{author}%'"

    def _build_recent_sql(self, query: str) -> str:
        # Example: get recent papers
        return "SELECT * FROM papers ORDER BY date_published DESC LIMIT 10"
