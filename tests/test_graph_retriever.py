import pytest
from retrievers.graph_retriever import GraphRetriever

class DummySession:
    def run(self, cypher_query, parameters=None):
        class DummyResult:
            def data(self):
                return {'paper': 'p1', 'author': 'a1'}
        return [DummyResult()]
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyDriver:
    def session(self):
        return DummySession()
    def close(self):
        pass

def test_graph_retriever(monkeypatch):
    monkeypatch.setattr('neo4j.GraphDatabase.driver', lambda uri, auth: DummyDriver())
    retriever = GraphRetriever('bolt://dummy', 'user', 'pass')
    results = retriever.retrieve('MATCH (n) RETURN n')
    assert isinstance(results, list)
    assert results[0]['paper'] == 'p1'
