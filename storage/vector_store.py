import os
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """Initialize vector store with embedding model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path or f"faiss_index_{model_name.replace('/', '_')}.idx"
        self.metadata_path = metadata_path or f"faiss_metadata_{model_name.replace('/', '_')}.pkl"
        
        # Initialize or load index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        if not texts:
            logger.warning("No texts provided to add to vector store")
            return
            
        logger.info(f"Adding {len(texts)} documents to vector store")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Vector store now contains {len(self.metadata)} documents")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if len(self.metadata) == 0:
            logger.warning("Vector store is empty, no results to return")
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata) and idx != -1:
                result = self.metadata[idx].copy()
                result['score'] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                results.append(result)
                
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        logger.info(f"Saving vector store to {self.index_path} and {self.metadata_path}")
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        logger.info("Vector store saved successfully")
    
    def load(self) -> None:
        """Load index and metadata from disk."""
        logger.info(f"Loading vector store from {self.index_path} and {self.metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        logger.info(f"Loaded vector store with {len(self.metadata)} documents")