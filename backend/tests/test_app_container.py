from app.app_container import (
    get_chat_service,
    get_document_indexer,
    get_rag_tool,
    get_retrieval_runtime,
    get_retriever,
)
from app.agents.research_agent import ResearchAgent
from app.rag.rag_tool import RAGTool
from app.retrieval.indexer import DocumentIndexer
from app.retrieval.retriever import Retriever


def setup_function() -> None:
    get_retrieval_runtime.cache_clear()
    get_document_indexer.cache_clear()
    get_rag_tool.cache_clear()
    get_retriever.cache_clear()
    get_chat_service.cache_clear()


def test_retrieval_runtime_is_shared_across_factories():
    indexer = get_document_indexer()
    rag_tool = get_rag_tool()
    retriever = get_retriever()

    assert isinstance(indexer, DocumentIndexer)
    assert isinstance(rag_tool, RAGTool)
    assert isinstance(retriever, Retriever)

    assert indexer.store is retriever._store
    assert rag_tool.retriever is retriever


def test_chat_service_uses_rag_tool_from_composition_root():
    service = get_chat_service()

    agent = service.agents[next(iter(service.agents))]
    assert isinstance(agent, ResearchAgent)
    assert agent.rag_tool is get_rag_tool()
