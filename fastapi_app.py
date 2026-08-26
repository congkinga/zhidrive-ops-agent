#!/usr/bin/env python3
"""FastAPI upgrade for the intelligent-driving operations Agent.

This service keeps the original stdlib server intact while adding:
- FastAPI REST endpoints
- SQLAlchemy/SQLite structured persistence
- LangGraph-based Agent workflow
- LangChain-compatible DeepSeek/OpenAI client
- SSE streaming responses
- Redis-ready metrics and state layer
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

import agent_server
from agent_server import (
    DOC_FILES,
    analyze_feedback,
    answer,
    generate_report,
    rag_search,
    run_evaluation,
    system_metrics,
)
from storage import load_cases
from vector_rag import VectorRAG
from retrieval_hybrid import HybridRetriever
from ops_analytics import (
    get_activities,
    get_content,
    get_funnel,
    get_overview,
    get_research,
    get_segments,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "ops.db"
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DB_PATH.as_posix()}",
)


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class CaseRecord(Base):
    __tablename__ = "ops_cases"

    id = Column(String(64), primary_key=True)
    input_text = Column(Text, nullable=False)
    scenario = Column(String(120), default="")
    observation = Column(Text, default="")
    severity = Column(String(12), default="")
    dimensions_json = Column(Text, default="[]")
    suggested_owner = Column(String(240), default="")
    short_term_action = Column(Text, default="")
    verification = Column(Text, default="")
    risk = Column(Text, default="")
    mode = Column(String(32), default="local")
    sources_json = Column(Text, default="[]")
    updated_at = Column(String(32), default="")


class ModelCallRecord(Base):
    __tablename__ = "model_calls"

    id = Column(String(64), primary_key=True)
    model = Column(String(80), default="")
    mode = Column(String(32), default="")
    latency_ms = Column(String(32), default="")
    user_content_chars = Column(String(32), default="")
    reply_chars = Column(String(32), default="")
    json_mode = Column(String(12), default="false")
    timestamp = Column(String(32), default="")


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    text: str


class RagSearchRequest(BaseModel):
    message: str
    top_k: int = 5


class GraphRunRequest(BaseModel):
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)


def load_dotenv() -> None:
    for name in (".env.local", ".env"):
        path = ROOT / name
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def parse_json_field(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def case_to_dict(case: CaseRecord) -> dict[str, Any]:
    analysis = {
        "scenario": case.scenario,
        "observation": case.observation,
        "severity": case.severity,
        "dimensions": parse_json_field(case.dimensions_json, []),
        "suggested_owner": case.suggested_owner,
        "short_term_action": case.short_term_action,
        "verification": case.verification,
        "risk": case.risk,
    }
    return {
        "id": case.id,
        "input": case.input_text,
        "analysis": analysis,
        "mode": case.mode,
        "sources": parse_json_field(case.sources_json, []),
        "updated_at": case.updated_at,
    }


def sync_cases_from_json() -> None:
    session = SessionLocal()
    try:
        count = session.query(CaseRecord).count()
        if count > 0:
            return
        for item in load_cases():
            analysis = item.get("analysis") or {}
            dimensions = analysis.get("dimensions") or []
            sources = item.get("sources") or []
            session.add(
                CaseRecord(
                    id=item.get("id") or f"case-{int(time.time() * 1000)}",
                    input_text=item.get("input", ""),
                    scenario=analysis.get("scenario", ""),
                    observation=analysis.get("observation", ""),
                    severity=analysis.get("severity", ""),
                    dimensions_json=json.dumps(dimensions, ensure_ascii=False),
                    suggested_owner=analysis.get("suggested_owner", ""),
                    short_term_action=analysis.get("short_term_action", ""),
                    verification=analysis.get("verification", ""),
                    risk=analysis.get("risk", ""),
                    mode=item.get("mode", "local"),
                    sources_json=json.dumps(sources, ensure_ascii=False),
                    updated_at=item.get("updated_at", ""),
                )
            )
        session.commit()
    finally:
        session.close()


def save_case_to_db(result: dict[str, Any]) -> str:
    analysis = result.get("analysis") or {}
    case_id = result.get("case_id") or f"case-{int(time.time() * 1000)}"
    session = SessionLocal()
    try:
        record = session.get(CaseRecord, case_id)
        if not record:
            record = CaseRecord(id=case_id, input_text=result.get("input", ""))
            session.add(record)
        record.input_text = result.get("input", record.input_text)
        record.scenario = analysis.get("scenario", "")
        record.observation = analysis.get("observation", "")
        record.severity = analysis.get("severity", "")
        record.dimensions_json = json.dumps(analysis.get("dimensions") or [], ensure_ascii=False)
        record.suggested_owner = analysis.get("suggested_owner", "")
        record.short_term_action = analysis.get("short_term_action", "")
        record.verification = analysis.get("verification", "")
        record.risk = analysis.get("risk", "")
        record.mode = result.get("mode", "local")
        record.sources_json = json.dumps(result.get("sources") or [], ensure_ascii=False)
        record.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        session.commit()
        return case_id
    finally:
        session.close()


def all_cases_from_db() -> list[dict[str, Any]]:
    session = SessionLocal()
    try:
        records = session.query(CaseRecord).order_by(CaseRecord.updated_at.desc()).limit(200).all()
        return [case_to_dict(record) for record in records]
    finally:
        session.close()


def chunk_text(text: str, size: int = 48) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]


def build_llm():
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    if os.getenv("DEEPSEEK_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            temperature=0.2,
        )
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            temperature=0.2,
        )
    return None


def build_graph():
    try:
        from langgraph.graph import END, START, StateGraph
        from typing import TypedDict
    except Exception:
        return None

    class OpsState(TypedDict, total=False):
        query: str
        history: list[dict[str, str]]
        sources: list[str]
        context: str
        analysis: dict[str, Any]
        mode: str
        reply: str

    def retrieve_node(state: OpsState) -> dict[str, Any]:
        query = state.get("query", "")
        results = rag_search(query, top_k=4)
        return {
            "sources": [item.get("source", "") for item in results],
            "context": json.dumps(results, ensure_ascii=False),
        }

    def analyze_node(state: OpsState) -> dict[str, Any]:
        query = state.get("query", "")
        result = analyze_feedback(query)
        analysis = result.get("analysis") or {}
        save_case_to_db({"input": query, **result})
        return {
            "analysis": analysis,
            "mode": result.get("mode", "local"),
        }

    def respond_node(state: OpsState) -> dict[str, Any]:
        result = answer(state.get("query", ""), history=state.get("history") or [])
        return {
            "reply": result.get("reply", ""),
            "sources": result.get("sources", state.get("sources", [])),
        }

    graph = StateGraph(OpsState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def create_app() -> FastAPI:
    load_dotenv()
    Base.metadata.create_all(bind=engine)
    sync_cases_from_json()
    agent_server.load_dotenv()
    vector_rag = VectorRAG(ROOT, DOC_FILES)
    agent_server.VECTOR_RAG = vector_rag
    hybrid_retriever = HybridRetriever(
        ROOT,
        DOC_FILES,
        vector=vector_rag,
        lexical=agent_server.RAG,
    )

    app = FastAPI(title="ZHIDRIVE OPS AGENT", version="0.3")
    graph = build_graph()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "zhidrive-ops-fastapi",
            "version": "0.3",
            "vector_rag": bool(agent_server.VECTOR_RAG and agent_server.VECTOR_RAG.available),
            "langgraph": graph is not None,
            "langchain": build_llm() is not None,
            "hybrid_retriever": hybrid_retriever is not None,
            "ops_analytics": True,
        }

    @app.get("/api/cases")
    def get_cases() -> dict[str, Any]:
        return {"cases": all_cases_from_db()}

    @app.get("/api/v1/cases")
    def get_cases_v1() -> dict[str, Any]:
        return {"cases": all_cases_from_db()}

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        return answer(payload.message, history=payload.history[-8:])

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest) -> StreamingResponse:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        def event_stream() -> Iterator[str]:
            result = answer(payload.message, history=payload.history[-8:])
            reply = result.get("reply", "")
            sources = result.get("sources", [])
            for chunk in chunk_text(reply):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/analyze")
    async def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        result = analyze_feedback(payload.text)
        result["case_id"] = save_case_to_db({"input": payload.text, **result})
        return result

    @app.post("/api/rag/search")
    async def search_rag(payload: RagSearchRequest) -> dict[str, Any]:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        return {
            "query": payload.message,
            "results": rag_search(payload.message, top_k=payload.top_k),
        }

    @app.post("/api/rag/hybrid")
    async def search_hybrid(payload: RagSearchRequest) -> dict[str, Any]:
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        return {
            "query": payload.message,
            "results": hybrid_retriever.search(payload.message, top_k=payload.top_k),
        }

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        return system_metrics()

    @app.get("/api/ops/overview")
    async def ops_overview() -> dict[str, Any]:
        return get_overview()

    @app.get("/api/ops/funnel")
    async def ops_funnel() -> dict[str, Any]:
        return get_funnel()

    @app.get("/api/ops/segments")
    async def ops_segments() -> dict[str, Any]:
        return get_segments()

    @app.get("/api/ops/activities")
    async def ops_activities() -> dict[str, Any]:
        return get_activities()

    @app.get("/api/ops/content")
    async def ops_content() -> dict[str, Any]:
        return get_content()

    @app.get("/api/ops/research")
    async def ops_research() -> dict[str, Any]:
        return get_research()

    @app.get("/api/report")
    async def report() -> dict[str, Any]:
        return generate_report()

    @app.post("/api/eval")
    async def evaluate() -> dict[str, Any]:
        return run_evaluation()

    @app.post("/api/v1/graph/run")
    async def run_graph(payload: GraphRunRequest) -> dict[str, Any]:
        if graph is None:
            raise HTTPException(status_code=503, detail="langgraph unavailable")
        initial: dict[str, Any] = {
            "query": payload.query,
            "history": payload.history[-8:],
        }
        return graph.invoke(initial)

    app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",
        port=int(os.getenv("FASTAPI_PORT", "8766")),
        reload=False,
    )
