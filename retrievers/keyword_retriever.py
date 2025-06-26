from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional

class KeywordRetriever:
    def __init__(self, es_host: str, index_name: str):
        self.es = Elasticsearch(es_host)
        self.index_name = index_name

    def retrieve(self, query: str, top_k: int = 5, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform a keyword search using Elasticsearch.
        """
        search_fields = fields or ["title", "abstract", "full_text"]
        es_query = {
            "multi_match": {
                "query": query,
                "fields": search_fields,
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        response = self.es.search(index=self.index_name, query=es_query, size=top_k)
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            results.append({
                "score": hit.get("_score"),
                "id": hit.get("_id"),
                "source": hit.get("_source")
            })
        return results
