import pytest
from unittest.mock import patch
from retrievers.database_retriever import DatabaseRetriever

class DummyResult:
    def __init__(self):
        self._rows = [(1, 'Test Paper')]
        self._keys = ['id', 'title']
        
    def fetchall(self):
        return self._rows
        
    def keys(self):
        return self._keys

class DummyConnection:
    def execute(self, sql, params=None):
        return DummyResult()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyEngine:
    def connect(self):
        return DummyConnection()

@patch('retrievers.database_retriever.create_engine')
def test_database_retriever(mock_create_engine):
    # Configure the mock to return our dummy engine
    mock_create_engine.return_value = DummyEngine()
    
    # Create retriever with a valid SQLAlchemy URL format
    retriever = DatabaseRetriever('postgresql://user:pass@localhost/testdb')
    
    # Execute the retrieve method
    results = retriever.retrieve('SELECT * FROM papers')
    
    # Assertions
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['title'] == 'Test Paper'
    assert results[0]['id'] == 1
    
    # Verify that create_engine was called with the correct URL
    mock_create_engine.assert_called_once_with('postgresql://user:pass@localhost/testdb')