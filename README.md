# RAG Research Assistant

## Overview

The RAG (Retrieval-Augmented Generation) Research Assistant is an end-to-end, full-stack AI system designed to help users query, explore, and synthesize knowledge from arXiv academic papers. It leverages multiple retrieval backends (vector search, keyword search, knowledge graph, and relational database) and combines them with a powerful language model to deliver rich, context-aware answers to research questions.

---

## Problem Statement

Academic research is growing rapidly, making it difficult for researchers and practitioners to efficiently find, connect, and synthesize relevant information from vast literature. Traditional keyword search is limited in semantic understanding, while pure LLMs lack factual grounding. This project solves these challenges by combining:
- **Semantic search** (vector retrieval)
- **Keyword search** (Elasticsearch)
- **Knowledge graph queries** (Neo4j)
- **Structured data queries** (PostgreSQL)
- **Generative synthesis** (LLM, e.g., Gemini)

---

## Tech Stack

- **Python 3.10+**
- **FastAPI**: REST API and web interface
- **FAISS**: Vector similarity search
- **Elasticsearch**: Keyword/document search
- **Neo4j**: Knowledge graph storage and queries
- **PostgreSQL**: Relational data storage
- **Sentence Transformers**: Embedding generation
- **LangChain / LangGraph**: Orchestration and workflow
- **Gemini (Google Generative AI)**: LLM for answer synthesis
- **Docker Compose**: Service orchestration
- **Streamlit/Gradio**: (Optional) Richer UI

---

## System Pipeline

1. **Data Ingestion**
    - Download arXiv metadata (title, abstract, arxiv_id) using `build_arxiv_faiss.py`.
    - Generate embeddings for abstracts and build a FAISS index and metadata file.
    - Ingest metadata into Elasticsearch, Neo4j, and PostgreSQL using `ingest_all_backends.py`.

2. **Backend Services**
    - Run Elasticsearch, Neo4j, and PostgreSQL via Docker Compose.
    - Each backend enables a different retrieval strategy.

3. **Query Orchestration**
    - User submits a research question via the web UI or API.
    - The orchestrator routes the query to vector, keyword, graph, and database retrievers.
    - Results are deduplicated, ranked, and formatted.
    - The LLM synthesizes a final answer using the retrieved context.

4. **User Interface**
    - Simple web UI (FastAPI HTML) for queries.
    - (Optional) Streamlit or Gradio for advanced visualization.

---

## Getting Started

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd RAG_research_agent
```

### 2. Set Up Python Environment
```sh
python -m venv .venv
.venv\Scripts\activate  # On Windows
# Or: source .venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### 3. Build the FAISS Index and Metadata
```sh
python build_arxiv_faiss.py
```

### 4. Start Backend Services (Docker Compose)
```sh
docker-compose up -d
```

### 5. Ingest Data into All Backends
```sh
python ingest_all_backends.py
```

### 6. Run the FastAPI App
```sh
uvicorn app:app --reload
```
Visit [http://localhost:8000](http://localhost:8000) for the web UI, or [http://localhost:8000/docs](http://localhost:8000/docs) for the API docs.

---

## Customization & Extending the Project

- **Change arXiv Query/Category**: Edit `QUERY` in `build_arxiv_faiss.py` to target different fields or categories.
- **Increase Data Volume**: Adjust `MAX_RESULTS` in `build_arxiv_faiss.py`.
- **Add More Metadata**: Extend the ingestion scripts to include authors, citations, or full text.
- **Switch Embedding Model**: Change `EMBEDDING_MODEL` in `build_arxiv_faiss.py`.
- **UI Enhancements**: Replace the default HTML UI with Streamlit or Gradio for richer interaction.
- **Backend Scaling**: Use managed services or scale Docker containers for production.
- **Add New Retrieval Strategies**: Implement new retrievers in the `retrievers/` folder and register them in the orchestrator.

---

## Folder Structure

```
RAG_research_agent/
├── app.py                  # FastAPI app and web UI
├── build_arxiv_faiss.py    # arXiv-to-FAISS pipeline
├── ingest_all_backends.py  # Ingest into Elasticsearch, Neo4j, PostgreSQL
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── retrievers/             # Retrieval modules (vector, keyword, graph, db)
├── storage/                # Storage logic
├── data/                   # FAISS index, metadata, and other artifacts
├── tests/                  # Unit tests
├── synthesis.py            # Deduplication, ranking, formatting
├── orchestrator.py         # Query routing and orchestration
├── langgraph_workflow.py   # LangGraph workflow
└── ...
```

---

## Troubleshooting

- **Elasticsearch/Neo4j/Postgres not connecting?**
    - Ensure Docker containers are running and healthy.
    - Check ports (9200, 7687, 5432) are not blocked.
- **FAISS index or metadata missing?**
    - Run `build_arxiv_faiss.py` first.
- **Module import errors?**
    - Activate your virtual environment and check `PYTHONPATH`.
- **Out of memory?**
    - Lower `MAX_RESULTS` in `build_arxiv_faiss.py`.

---

## License

This project is for educational and research purposes. Please check the arXiv terms of use and the licenses of all dependencies before deploying in production.
