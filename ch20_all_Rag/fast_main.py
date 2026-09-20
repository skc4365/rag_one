"""RAG 챗봇용 FastAPI 백엔드."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

try:
    # 프로젝트 루트에서: uvicorn ch20_all_Rag.fast_main:app
    from .rag import ask
except ImportError:
    # ch20_all_Rag 폴더에서: uvicorn fast_main:app
    from rag import ask


app = FastAPI(
    title="AI 교육과정 RAG 챗봇 API",
    description="myclass.txt 기반 검색 증강 생성(RAG) API",
    version="1.0.0",
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class SourceResponse(BaseModel):
    section: str
    source: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "AI 교육과정 RAG 챗봇 API가 실행 중입니다.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        history = [message.model_dump() for message in request.history]
        result = await run_in_threadpool(ask, request.question, history)
        return ChatResponse(**result)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"RAG 답변 생성 중 오류가 발생했습니다: {exc}",
        ) from exc
