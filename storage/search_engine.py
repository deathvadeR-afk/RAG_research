import os
import logging
from typing import Dict, Any, List, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, hosts: Optional[List[str]] = None, index_name: str = "research_papers"):
        """Initialize connection to Elasticsearch."""
        self.hosts = hosts or [os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")]
        self.index_name = index_name
        
        logger.info(f"Connecting to Elasticsearch at {self.hosts}")
        self.es = Elasticsearch(self.hosts)
        
        # Create index if it doesn't exist
        self._create_index()
        
    def _create_index(self) -> None:
        """Create index with appropriate mappings if it doesn't exist."""
        if not self.es.indices.exists(index=self.index_name):
            logger.info(f"Creating Elasticsearch index: {self.index_name}")
            
            mappings = {
                "mappings": {
                    "properties": {
                        "arxiv_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "english",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "abstract": {"type": "text", "analyzer": "english"},
                        "full_text": {"type": "text", "analyzer": "english"},
                        "authors": {
                            "type": "nested",
                            "properties": {
                                "full_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                                "affiliation": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
                            }
                        },
                        "publication_date": {"type": "date"},
                        "categories": {"type": "keyword"},
                        "primary_category": {"type": "keyword"},
                        "topics": {"type": "keyword"}
                    }
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "english": {
                                "tokenizer": "standard",
                                "filter": ["lowercase", "english_stop", "english_stemmer"]
                            }
                        },
                        "filter": {
                            "english_stop": {
                                "type": "stop",
                                "stopwords": "_english_"
                            },
                            "english_stemmer": {
                                "type": "stemmer",
                                "language": "english"
                            }
                        }
                    }
                }
            }
            
            self.es.indices.create(index=self.index_name, body=mappings)
            logger.info(f"Created index {self.index_name} with mappings")
    
    def index_paper(self, paper: Dict[str, Any]) -> None:
        """Index a single paper document."""
        doc_id = paper['arxiv_id']
        
        # Prepare document
        document = {
            'arxiv_id': paper['arxiv_id'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'authors': paper.get('authors', []),
            'publication_date': paper['publication_date'],
            'categories': paper.get('categories', []),
            'primary_category': paper['primary_category'],
            'topics': paper.get('topics', [])
        }
        
        # Add full text if available
        if 'full_text' in paper:
            document['full_text'] = paper['full_text']
        
        # Index document
        self.es.index(index=self.index_name, id=doc_id, body=document)
        logger.info(f"Indexed paper {doc_id}")
    
    def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> None:
        """Index multiple papers in bulk."""
        if not papers:
            logger.warning("No papers provided for bulk indexing")
            return
            
        logger.info(f"Bulk indexing {len(papers)} papers")
        
        # Prepare actions
        actions = []
        for paper in papers:
            doc_id = paper['arxiv_id']
            
            # Prepare document
            document = {
                '_index': self.index_name,
                '_id': doc_id,
                '_source': {
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'authors': paper.get('authors', []),
                    'publication_date': paper['publication_date'],
                    'categories': paper.get('categories', []),
                    'primary_category': paper['primary_category'],
                    'topics': paper.get('topics', [])
                }
            }
            
            # Add full text if available
            if 'full_text' in paper:
                document['_source']['full_text'] = paper['full_text']
                
            actions.append(document)
        
        # Execute bulk indexing
        success, failed = bulk(self.es, actions, refresh=True)
        logger.info(f"Bulk indexed {success} papers, {failed} failed")

    def search(self, query: str, fields: List[str] = None, filters: Dict[str, Any] = None, size: int = 10) -> List[Dict[str, Any]]:
        """Search for papers matching the query."""
        if fields is None:
            fields = ["title^3", "abstract^2", "full_text"]
            
        # Build query
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": fields,
                                "type": "best_fields",
                                "operator": "and"
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "abstract": {},
                    "full_text": {}
                },
                "pre_tags": ["<strong>"],
                "post_tags": ["</strong>"]
            },
            "size": size
        }
        
        # Add filters if provided
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    search_query["query"]["bool"]["filter"] = search_query["query"]["bool"].get("filter", [])
                    search_query["query"]["bool"]["filter"].append({"terms": {field: value}})
                else:
                    search_query["query"]["bool"]["filter"] = search_query["query"]["bool"].get("filter", [])
                    search_query["query"]["bool"]["filter"].append({"term": {field: value}})
        
        # Execute search
        logger.info(f"Searching for '{query}' in fields {fields}")
        response = self.es.search(index=self.index_name, body=search_query)
        
        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            result = hit["_source"]
            result["score"] = hit["_score"]
            result["highlights"] = hit.get("highlight", {})
            results.append(result)
            
        logger.info(f"Found {len(results)} results for query '{query}'")
        return results
    
    def search_by_author(self, author_name: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search for papers by a specific author."""
        query = {
            "query": {
                "nested": {
                    "path": "authors",
                    "query": {
                        "match": {
                            "authors.full_name": author_name
                        }
                    }
                }
            },
            "size": size
        }
        
        logger.info(f"Searching for papers by author '{author_name}'")
        response = self.es.search(index=self.index_name, body=query)
        
        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            result = hit["_source"]
            result["score"] = hit["_score"]
            results.append(result)
            
        logger.info(f"Found {len(results)} papers by author '{author_name}'")
        return results
    
    def search_by_topic(self, topic: str, size: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on a specific topic."""
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"topics": topic}},
                        {"term": {"categories": topic}},
                        {"term": {"primary_category": topic}}
                    ]
                }
            },
            "size": size
        }
        
        logger.info(f"Searching for papers on topic '{topic}'")
        response = self.es.search(index=self.index_name, body=query)
        
        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            result = hit["_source"]
            result["score"] = hit["_score"]
            results.append(result)
            
        logger.info(f"Found {len(results)} papers on topic '{topic}'")
        return results
    
    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific paper by arXiv ID."""
        try:
            response = self.es.get(index=self.index_name, id=arxiv_id)
            return response["_source"]
        except Exception as e:
            logger.error(f"Error retrieving paper {arxiv_id}: {e}")
            return None   
