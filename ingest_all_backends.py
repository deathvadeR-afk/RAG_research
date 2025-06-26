"""
Ingest arXiv metadata into Elasticsearch, Neo4j, and PostgreSQL.
Assumes you have already run build_arxiv_faiss.py and have data/faiss_meta.json.
"""
import json
import os
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import execute_batch

# --- Config ---
META_PATH = "data/faiss_meta.json"
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "papers")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
PG_URL = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/research")

# --- Load metadata ---
with open(META_PATH, encoding="utf-8") as f:
    papers = json.load(f)

# --- Elasticsearch Ingestion ---
print("[1/3] Indexing papers into Elasticsearch...")
es = Elasticsearch(ES_HOST)
if not es.indices.exists(index=ES_INDEX):
    es.indices.create(index=ES_INDEX, body={
        "mappings": {"properties": {"title": {"type": "text"}, "abstract": {"type": "text"}, "arxiv_id": {"type": "keyword"}}}
    })
for paper in papers:
    es.index(index=ES_INDEX, id=paper["arxiv_id"], document=paper)
print(f"Indexed {len(papers)} papers into Elasticsearch index '{ES_INDEX}'.")

# --- Neo4j Ingestion ---
print("[2/3] Creating nodes in Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as session:
    for paper in papers:
        session.run(
            "MERGE (p:Paper {arxiv_id: $arxiv_id}) SET p.title = $title, p.abstract = $abstract",
            {"arxiv_id": paper["arxiv_id"], "title": paper["title"], "abstract": paper["abstract"]}
        )
print(f"Created {len(papers)} Paper nodes in Neo4j.")
driver.close()

# --- PostgreSQL Ingestion ---
print("[3/3] Inserting papers into PostgreSQL...")
import urllib.parse as up
up.uses_netloc.append("postgres")
url = up.urlparse(PG_URL)
conn = psycopg2.connect(
    database=url.path[1:], user=url.username, password=url.password, host=url.hostname, port=url.port
)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT
)
""")
execute_batch(cur, "INSERT INTO papers (arxiv_id, title, abstract) VALUES (%s, %s, %s) ON CONFLICT (arxiv_id) DO NOTHING", [(p["arxiv_id"], p["title"], p["abstract"]) for p in papers])
conn.commit()
cur.close()
conn.close()
print(f"Inserted {len(papers)} papers into PostgreSQL table 'papers'.")

print("All backends ingested successfully!")
