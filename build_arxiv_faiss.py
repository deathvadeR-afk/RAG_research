# build_arxiv_faiss.py
"""
Automate the full arXiv-to-FAISS pipeline:
1. Download arXiv metadata (title, abstract, arxiv_id) for a given query/category.
2. Generate embeddings for abstracts using Sentence Transformers.
3. Build and save a FAISS index and metadata file.
"""
import arxiv
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Config ---
QUERY = "cat:cs.AI"  # Change to your desired arXiv category or query
MAX_RESULTS = 500     # Number of papers to fetch
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "faiss_meta.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)

print(f"[1/3] Downloading arXiv papers for query: {QUERY} (max {MAX_RESULTS})...")
search = arxiv.Search(query=QUERY, max_results=MAX_RESULTS)
papers = []
for result in search.results():
    papers.append({
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "abstract": result.summary.replace("\n", " ").strip()
    })
print(f"Downloaded {len(papers)} papers.")

print("[2/3] Generating embeddings for abstracts...")
model = SentenceTransformer(EMBEDDING_MODEL)
abstracts = [p["abstract"] for p in papers]
embeddings = model.encode(abstracts, show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")

print("[3/3] Building and saving FAISS index and metadata...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"Done! FAISS index: {INDEX_PATH}\nMetadata: {META_PATH}\nPapers indexed: {len(papers)}")
