import pytest
from retrievers.vector_retriever import VectorRetriever
import numpy as np

def test_vector_retriever_init(monkeypatch):
    class DummyIndex:
        def search(self, x, k):
            return np.array([[0.9, 0.8]]), np.array([[1, 2]])
    monkeypatch.setattr('faiss.read_index', lambda path: DummyIndex())
    monkeypatch.setattr('sentence_transformers.SentenceTransformer', lambda name: type('DummyModel', (), {'encode': lambda self, x: np.zeros((1, 768))})())
    retriever = VectorRetriever('dummy.index')
    assert retriever.index is not None
    assert retriever.model is not None

def test_vector_retriever_retrieve(monkeypatch):
    class DummyIndex:
        def search(self, x, k):
            return np.array([[0.9, 0.8]]), np.array([[1, 2]])
    monkeypatch.setattr('faiss.read_index', lambda path: DummyIndex())
    monkeypatch.setattr('sentence_transformers.SentenceTransformer', lambda name: type('DummyModel', (), {'encode': lambda self, x: np.zeros((1, 768))})())
    retriever = VectorRetriever('dummy.index')
    retriever.metadata = {1: {'title': 'Test Paper 1'}, 2: {'title': 'Test Paper 2'}}
    results = retriever.retrieve('test', top_k=2)
    assert len(results) == 2
    assert results[0]['metadata']['title'] == 'Test Paper 1'
