from orchestrator import Orchestrator
from synthesis import deduplicate_results, rank_results, format_for_generation
import os
import sys

def get_config():
    # Example config, replace with your actual paths and credentials
    return {
        'vector_cfg': {
            'index_path': os.getenv('FAISS_INDEX_PATH', 'data/faiss.index'),
            'metadata_path': os.getenv('FAISS_META_PATH', 'data/faiss_meta.json'),
        },
        'graph_cfg': {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password'),
        },
        'db_cfg': {
            'db_url': os.getenv('DB_URL', 'postgresql://user:password@localhost:5432/research'),
        },
        'keyword_cfg': {
            'es_host': os.getenv('ES_HOST', 'http://localhost:9200'),
            'index_name': os.getenv('ES_INDEX', 'papers'),
        },
    }

def main():
    config = get_config()
    orchestrator = Orchestrator(**config)
    print("Research Assistant CLI. Type your query (or 'exit' to quit):")
    while True:
        query = input('> ').strip()
        if query.lower() in ('exit', 'quit'):
            break
        # Simple heuristics for query type
        if query.lower().startswith('author:'):
            query_type = 'author'
        elif query.lower().startswith('recent'):
            query_type = 'recent'
        else:
            query_type = None
        results = orchestrator.process_query(query, query_type=query_type)
        deduped = deduplicate_results(results)
        ranked = rank_results(deduped)
        context = format_for_generation(ranked)
        print("\nTop Results:\n")
        print(context)
        print("\n---\n")

if __name__ == '__main__':
    main()
