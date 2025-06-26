from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List
from typing_extensions import TypedDict

from retrievers.vector_retriever import VectorRetriever
from retrievers.graph_retriever import GraphRetriever
from retrievers.database_retriever import DatabaseRetriever
from retrievers.keyword_retriever import KeywordRetriever
from synthesis import deduplicate_results, rank_results, format_for_generation

# --- LangChain Retriever Wrappers ---
class VectorLC(BaseRetriever):
    def __init__(self, retriever: VectorRetriever):
        super().__init__()
        self.retriever = retriever
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        results = self.retriever.retrieve(query, top_k=5)
        return [Document(page_content=str(r['metadata']), metadata=r['metadata']) for r in results]

class KeywordLC(BaseRetriever):
    def __init__(self, retriever: KeywordRetriever):
        super().__init__()
        self.retriever = retriever
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        results = self.retriever.retrieve(query, top_k=5)
        return [Document(page_content=str(r['source']), metadata=r['source']) for r in results]

# --- LangGraph State ---
class ResearchState(TypedDict):
    query: str
    vector_results: List[Document]
    keyword_results: List[Document]
    answer: str

# --- LangGraph Workflow ---
def build_langgraph_workflow(vector_cfg, keyword_cfg):
    # Initialize retrievers
    vector = VectorRetriever(**vector_cfg)
    keyword = KeywordRetriever(**keyword_cfg)
    vector_lc = VectorLC(vector)
    keyword_lc = KeywordLC(keyword)
    
    # LLM (Gemini)
    llm = GoogleGenerativeAI(model="gemini-pro")
    
    # RetrievalQA chains
    vector_chain = RetrievalQA.from_chain_type(llm, retriever=vector_lc)
    keyword_chain = RetrievalQA.from_chain_type(llm, retriever=keyword_lc)
    
    # LangGraph StateGraph
    graph = StateGraph(ResearchState)
    
    def vector_node(state: ResearchState) -> ResearchState:
        docs = vector_lc._get_relevant_documents(state["query"])
        return {
            **state,
            "vector_results": docs
        }
    
    def keyword_node(state: ResearchState) -> ResearchState:
        docs = keyword_lc._get_relevant_documents(state["query"])
        return {
            **state,
            "keyword_results": docs
        }
    
    def synthesis_node(state: ResearchState) -> ResearchState:
        # Combine and format results
        all_docs = state.get("vector_results", []) + state.get("keyword_results", [])
        context = "\n".join([d.page_content for d in all_docs])
        
        # Use invoke method instead of direct call
        response = llm.invoke(context + "\n\nAnswer the following question: " + state["query"])
        
        return {
            **state,
            "answer": response
        }
    
    # Add nodes to graph
    graph.add_node("vector_node", vector_node)
    graph.add_node("keyword_node", keyword_node)
    graph.add_node("synthesis_node", synthesis_node)
    
    # Add edges
    graph.add_edge(START, "vector_node")
    graph.add_edge("vector_node", "keyword_node")
    graph.add_edge("keyword_node", "synthesis_node")
    graph.add_edge("synthesis_node", END)
    
    # Compile and return the graph
    return graph.compile()