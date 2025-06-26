import pytest
from unittest.mock import patch
from retrievers.keyword_retriever import KeywordRetriever

class DummyES:
    def search(self, index, query, size):
        return {
            'hits': {
                'hits': [
                    {'_score': 1.0, '_id': '1', '_source': {'title': 'Test', 'abstract': 'A'}},
                    {'_score': 0.9, '_id': '2', '_source': {'title': 'Test2', 'abstract': 'B'}}
                ]
            }
        }

@patch('retrievers.keyword_retriever.Elasticsearch')
def test_keyword_retriever(mock_elasticsearch):
    # Configure the mock to return our dummy Elasticsearch client
    mock_elasticsearch.return_value = DummyES()
    
    # Create retriever with any host (it will be mocked)
    retriever = KeywordRetriever('http://localhost:9200', 'dummy_index')
    
    # Execute the retrieve method
    results = retriever.retrieve('test', top_k=2)
    
    # Assertions
    assert len(results) == 2
    assert results[0]['source']['title'] == 'Test'
    assert results[0]['source']['abstract'] == 'A'
    assert results[1]['source']['title'] == 'Test2'
    assert results[1]['source']['abstract'] == 'B'
    assert results[0]['score'] == 1.0
    assert results[1]['score'] == 0.9
    
    # Verify that Elasticsearch was called with the correct host
    mock_elasticsearch.assert_called_once_with('http://localhost:9200')