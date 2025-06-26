import pytest
from unittest.mock import patch, Mock
from langgraph_workflow import build_langgraph_workflow

class DummyVectorRetriever:
    def __init__(self, *args, **kwargs):
        pass
    def retrieve(self, query, top_k=5):
        return [{"metadata": {"title": "VectorDoc", "abstract": "Vector abstract."}}]

class DummyKeywordRetriever:
    def __init__(self, *args, **kwargs):
        pass
    def retrieve(self, query, top_k=5):
        return [{"source": {"title": "KeywordDoc", "abstract": "Keyword abstract."}}]

class DummyLLM:
    def __init__(self, *args, **kwargs):
        pass
    def invoke(self, prompt):
        return "This is a generated answer."

class DummyVectorLC:
    def __init__(self, retriever):
        self.retriever = retriever
    def _get_relevant_documents(self, query, run_manager=None):
        return [Mock(page_content="VectorDoc", metadata={"title": "VectorDoc"})]

class DummyKeywordLC:
    def __init__(self, retriever):
        self.retriever = retriever
    def _get_relevant_documents(self, query, run_manager=None):
        return [Mock(page_content="KeywordDoc", metadata={"title": "KeywordDoc"})]

@patch('langgraph_workflow.RetrievalQA')
@patch('langgraph_workflow.VectorLC', new=DummyVectorLC)
@patch('langgraph_workflow.KeywordLC', new=DummyKeywordLC)
@patch('langgraph_workflow.GoogleGenerativeAI', new=DummyLLM)
@patch('langgraph_workflow.KeywordRetriever', new=DummyKeywordRetriever)
@patch('langgraph_workflow.VectorRetriever', new=DummyVectorRetriever)
def test_langgraph_workflow(mock_retrievalqa):
    mock_retrievalqa.from_chain_type.return_value = Mock()
    vector_config = {'index_path': '/dummy/path/to/index.faiss'}
    keyword_config = {'host': 'http://localhost:9200', 'index': 'dummy_index'}
    graph = build_langgraph_workflow(vector_config, keyword_config)
    initial_state = {"query": "What is test?", "vector_results": [], "keyword_results": [], "answer": ""}
    result = graph.invoke(initial_state)
    assert isinstance(result, dict)
    assert 'answer' in result
    assert len(result['answer']) > 0
    assert "generated answer" in result['answer'].lower()

@patch('langgraph_workflow.RetrievalQA')
@patch('langgraph_workflow.VectorLC', new=DummyVectorLC)
@patch('langgraph_workflow.KeywordLC', new=DummyKeywordLC)
@patch('langgraph_workflow.GoogleGenerativeAI', new=DummyLLM)
@patch('langgraph_workflow.KeywordRetriever', new=DummyKeywordRetriever)
@patch('langgraph_workflow.VectorRetriever', new=DummyVectorRetriever)
def test_langgraph_workflow_comprehensive(mock_retrievalqa):
    mock_retrievalqa.from_chain_type.return_value = Mock()
    vector_config = {'index_path': './test_data/vector_index.faiss', 'metadata_path': './test_data/metadata.json', 'model_name': 'sentence-transformers/all-MiniLM-L6-v2'}
    keyword_config = {'host': 'http://localhost:9200', 'index': 'research_papers'}
    graph = build_langgraph_workflow(vector_config, keyword_config)
    initial_state = {"query": "What is machine learning?", "vector_results": [], "keyword_results": [], "answer": ""}
    result = graph.invoke(initial_state)
    assert isinstance(result, dict)
    assert 'answer' in result
    assert 'vector_results' in result
    assert 'keyword_results' in result
    assert len(result['answer']) > 0

@patch('langgraph_workflow.RetrievalQA')
@patch('langgraph_workflow.VectorLC', new=DummyVectorLC)
@patch('langgraph_workflow.KeywordLC', new=DummyKeywordLC)
@patch('langgraph_workflow.GoogleGenerativeAI', new=DummyLLM)
@patch('langgraph_workflow.KeywordRetriever', new=DummyKeywordRetriever)
@patch('langgraph_workflow.VectorRetriever', new=DummyVectorRetriever)
def test_langgraph_workflow_minimal_config(mock_retrievalqa):
    mock_retrievalqa.from_chain_type.return_value = Mock()
    vector_config = {'index_path': '/path/to/index.faiss'}
    keyword_config = {'host': 'localhost:9200', 'index': 'test'}
    graph = build_langgraph_workflow(vector_config, keyword_config)
    state = {"query": "test query"}
    result = graph.invoke(state)
    assert 'answer' in result
    assert len(result['answer']) > 0

@pytest.fixture
def mock_workflow_components():
    with patch('langgraph_workflow.RetrievalQA') as mock_retrievalqa, \
         patch('langgraph_workflow.VectorLC', new=DummyVectorLC), \
         patch('langgraph_workflow.KeywordLC', new=DummyKeywordLC), \
         patch('langgraph_workflow.GoogleGenerativeAI', new=DummyLLM), \
         patch('langgraph_workflow.KeywordRetriever', new=DummyKeywordRetriever), \
         patch('langgraph_workflow.VectorRetriever', new=DummyVectorRetriever):
        mock_retrievalqa.from_chain_type.return_value = Mock()
        yield True

def test_langgraph_workflow_with_fixture():
    with patch('langgraph_workflow.RetrievalQA') as mock_retrievalqa, \
         patch('langgraph_workflow.VectorLC', new=DummyVectorLC), \
         patch('langgraph_workflow.KeywordLC', new=DummyKeywordLC), \
         patch('langgraph_workflow.GoogleGenerativeAI', new=DummyLLM), \
         patch('langgraph_workflow.KeywordRetriever', new=DummyKeywordRetriever), \
         patch('langgraph_workflow.VectorRetriever', new=DummyVectorRetriever):
        mock_retrievalqa.from_chain_type.return_value = Mock()
        vector_config = {'index_path': './fixture_test_index.faiss', 'model_name': 'all-MiniLM-L6-v2'}
        keyword_config = {'host': 'localhost:9200', 'index': 'fixture_papers'}
        graph = build_langgraph_workflow(vector_config, keyword_config)
        state = {"query": "fixture test query"}
        result = graph.invoke(state)
        assert 'answer' in result
        assert "generated answer" in result['answer'].lower()