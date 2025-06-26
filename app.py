from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi import Request
from orchestrator import Orchestrator
from synthesis import deduplicate_results, rank_results, format_for_generation
import os

app = FastAPI(title="RAG Research Assistant API")

def get_config():
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

@app.on_event("startup")
def startup_event():
    global orchestrator
    orchestrator = Orchestrator(**get_config())

@app.get("/query")
def query_endpoint(q: str = Query(..., description="Your research question")):
    results = orchestrator.process_query(q)
    deduped = deduplicate_results(results)
    ranked = rank_results(deduped)
    context = format_for_generation(ranked)
    html_context = context.replace('\n', '<br>')
    return {"results": context, "results_html": html_context}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>arXiv Research Assistant</title>
        </head>
        <body>
            <h2>arXiv Research Assistant</h2>
            <form id="query-form">
                <input type="text" id="query" placeholder="Type your research question..." size="50"/>
                <button type="submit">Ask</button>
            </form>
            <div id="answer" style="margin-top:20px;"></div>
            <script>
                document.getElementById('query-form').onsubmit = async function(e) {
                    e.preventDefault();
                    const query = document.getElementById('query').value;
                    document.getElementById('answer').innerText = "Loading...";
                    const res = await fetch('/query?q=' + encodeURIComponent(query));
                    const data = await res.json();
                    if (data.results_html) {
                        document.getElementById('answer').innerHTML = data.results_html;
                    } else if (data.results) {
                        document.getElementById('answer').innerText = data.results;
                    } else {
                        document.getElementById('answer').innerText = JSON.stringify(data);
                    }
                }
            </script>
        </body>
    </html>
    """
