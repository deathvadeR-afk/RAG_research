from typing import List, Dict, Any
from collections import defaultdict
import hashlib

def deduplicate_results(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Deduplicate results from multiple retrievers based on document id or content hash.
    """
    seen = set()
    deduped = []
    for retriever, items in results.items():
        for item in items:
            # Try to use a unique id, else hash the content
            doc_id = item.get('id') or item.get('index')
            if not doc_id:
                doc_id = hashlib.md5(str(item).encode('utf-8')).hexdigest()
            if doc_id not in seen:
                seen.add(doc_id)
                deduped.append(item)
    return deduped

def rank_results(results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Rank results by score if available, else return as is.
    """
    return sorted(results, key=lambda x: -x.get('score', 0))[:top_k]

def format_for_generation(results: List[Dict[str, Any]]) -> str:
    """
    Format results into a string for prompt context.
    """
    formatted = []
    for i, item in enumerate(results, 1):
        meta = item.get('metadata') or item.get('source') or {}
        title = meta.get('title', '[No Title]')
        snippet = meta.get('abstract') or meta.get('summary') or str(meta)[:200]
        formatted.append(f"[{i}] {title}\n{snippet}\n")
    return '\n'.join(formatted)
