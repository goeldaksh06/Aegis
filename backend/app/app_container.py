from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

from app.events.bus import event_bus
from app.events.cost import cost_collector
from app.events.evaluation import evaluation_collector
from app.events.moderation import moderation_collector
from app.events.telemetry import telemetry_collector
from app.agents.router import agent_router
from app.agents.research_agent import ResearchAgent
from app.agents.analyst_agent import AnalystAgent
from app.agents.coder_agent import CoderAgent
from app.agents.document_agent import DocumentAgent
from app.agents.planner_agent import PlannerAgent
from app.api.chat_service import ChatService
from app.llm.service import LLMService
from app.models.schemas import AgentType
from app.rag.rag_tool import RAGTool
from app.retrieval.chunker import TextChunker
from app.retrieval.indexer import DocumentIndexer
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever
from app.retrieval.seed_documents import index_seed_documents


@dataclass(frozen=True)
class RetrievalRuntime:
    chunker: TextChunker
    embedding_service: EmbeddingService
    store: FAISSStore
    retriever: Retriever
    rag_tool: RAGTool
    indexer: DocumentIndexer


@lru_cache
def get_retrieval_runtime() -> RetrievalRuntime:
    chunker = TextChunker()
    embedding_service = EmbeddingService()
    store = FAISSStore(dimension=384)
    retriever = Retriever(
        embedding_service=embedding_service,
        store=store,
    )
    rag_tool = RAGTool(retriever=retriever)
    indexer = DocumentIndexer(
        chunker=chunker,
        embedding_service=embedding_service,
        store=store,
    )

    index_seed_documents(indexer, retriever)

    return RetrievalRuntime(
        chunker=chunker,
        embedding_service=embedding_service,
        store=store,
        retriever=retriever,
        rag_tool=rag_tool,
        indexer=indexer,
    )


@lru_cache
def get_document_indexer() -> DocumentIndexer:
    return get_retrieval_runtime().indexer


@lru_cache
def get_rag_tool() -> RAGTool:
    return get_retrieval_runtime().rag_tool


@lru_cache
def get_retriever() -> Retriever:
    return get_retrieval_runtime().retriever


@lru_cache
def get_chat_service() -> ChatService:
    llm_service = LLMService()
    research_agent = ResearchAgent(
        llm_service=llm_service,
        rag_tool=get_rag_tool(),
    )
    analyst_agent = AnalystAgent(
        llm_service=llm_service,
        rag_tool=get_rag_tool(),
    )
    coder_agent = CoderAgent(
        llm_service=llm_service,
        rag_tool=get_rag_tool(),
    )
    document_agent = DocumentAgent(
        llm_service=llm_service,
        rag_tool=get_rag_tool(),
    )
    planner_agent = PlannerAgent(
        llm_service=llm_service,
        rag_tool=get_rag_tool(),
    )

    event_bus.subscribe("request.received", telemetry_collector.handle)
    event_bus.subscribe("agent.selected", telemetry_collector.handle)
    event_bus.subscribe("request.completed", telemetry_collector.handle)
    event_bus.subscribe("request.failed", telemetry_collector.handle)
    event_bus.subscribe("request.completed", evaluation_collector.handle)
    event_bus.subscribe("request.completed", cost_collector.handle)
    event_bus.subscribe("request.completed", moderation_collector.handle)

    return ChatService(
        agent_router=agent_router,
        agents={
            AgentType.RESEARCH: research_agent,
            AgentType.ANALYST: analyst_agent,
            AgentType.CODER: coder_agent,
            AgentType.DOCUMENT: document_agent,
            AgentType.PLANNER: planner_agent,
        },
    )
