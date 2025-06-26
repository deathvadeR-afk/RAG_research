import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorRetriever:
    def __init__(self, index_path: str, metadata_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        index_path: Path to the FAISS index file
        metadata_path: Path to a numpy or json file mapping index ids to metadata (optional)
        model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            if metadata_path.endswith('.npy'):
                self.metadata = np.load(metadata_path, allow_pickle=True).item()
            elif metadata_path.endswith('.json'):
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns top_k most similar documents for the query.
        """
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype('float32'), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            meta = self.get_metadata(idx)
            results.append({
                'index': int(idx),
                'score': float(score),
                'metadata': meta
            })
        return results

    def get_metadata(self, idx: int) -> Any:
        if self.metadata is not None:
            if isinstance(self.metadata, dict):
                return self.metadata.get(str(idx)) or self.metadata.get(idx)
            elif isinstance(self.metadata, list):
                if 0 <= idx < len(self.metadata):
                    return self.metadata[idx]
        return None
