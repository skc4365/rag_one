"""myclass.txt를 지식 베이스로 사용하는 간단한 RAG 파이프라인."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "myclass.txt"

# 프로젝트 루트의 .env와 현재 폴더의 .env를 모두 지원합니다.
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=True)


class Source(TypedDict):
    section: str
    source: str
    content: str


class RAGResult(TypedDict):
    answer: str
    sources: list[Source]


def _load_documents(path: Path = SOURCE_PATH) -> list[Document]:
    """강의 안내문을 STEP 단위의 검색 문서로 변환합니다."""
    if not path.exists():
        raise FileNotFoundError(f"지식 문서를 찾을 수 없습니다: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"지식 문서가 비어 있습니다: {path}")

    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?=^STEP\s+\d+)", text, flags=re.MULTILINE)
        if chunk.strip()
    ]
    documents: list[Document] = []

    for index, chunk in enumerate(chunks, start=1):
        first_line = chunk.splitlines()[0].strip()
        match = re.match(r"(STEP\s+\d+)\s*·\s*(\d+h)", first_line)
        section = match.group(1) if match else f"SECTION {index:02d}"
        duration = match.group(2) if match else ""
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": path.name,
                    "section": section,
                    "duration": duration,
                },
            )
        )

    return documents


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "(이전 대화 없음)"

    role_names = {"user": "사용자", "assistant": "챗봇"}
    lines = []
    for message in history[-6:]:
        role = role_names.get(message.get("role", ""))
        content = message.get("content", "").strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(이전 대화 없음)"


class CourseRAG:
    """검색기와 답변 체인을 한 번만 생성해 재사용합니다."""

    def __init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.documents = _load_documents()
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self.vector_store = InMemoryVectorStore.from_documents(self.documents, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 AI 교육과정 안내 챗봇입니다. 반드시 제공된 문서만 근거로 "
                    "한국어로 정확하고 친절하게 답하세요. 문서에 없는 내용은 추측하지 말고 "
                    "'제공된 교육과정 문서에서는 확인할 수 없습니다.'라고 답하세요. "
                    "가능하면 관련 STEP과 교육 시간을 함께 알려주세요."
                    "\n\n[교육과정 문서]\n{context}",
                ),
                (
                    "human",
                    "[이전 대화]\n{history}\n\n[현재 질문]\n{question}",
                ),
            ]
        )
        model = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            temperature=0,
        )
        self.chain = prompt | model | StrOutputParser()

    def ask(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> RAGResult:
        question = question.strip()
        if not question:
            raise ValueError("질문을 입력해 주세요.")

        # 전체 과정/총 교육시간 질문에는 일부 STEP만 검색되면 계산이 틀릴 수 있습니다.
        aggregate_keywords = ("전체", "총", "합계", "몇 단계", "모든 단계")
        found = (
            self.documents
            if any(keyword in question for keyword in aggregate_keywords)
            else self.retriever.invoke(question)
        )
        context = "\n\n---\n\n".join(document.page_content for document in found)
        answer = self.chain.invoke(
            {
                "question": question,
                "history": _format_history(history),
                "context": context,
            }
        )

        sources: list[Source] = [
            {
                "section": str(document.metadata.get("section", "")),
                "source": str(document.metadata.get("source", SOURCE_PATH.name)),
                "content": document.page_content,
            }
            for document in found
        ]
        return {"answer": answer, "sources": sources}


@lru_cache(maxsize=1)
def get_rag() -> CourseRAG:
    return CourseRAG()


def ask(question: str, history: list[dict[str, str]] | None = None) -> RAGResult:
    """FastAPI와 CLI가 함께 사용하는 공개 함수입니다."""
    return get_rag().ask(question, history)


if __name__ == "__main__":
    result = ask("전체 교육 과정은 몇 단계이고, RAG는 어느 단계에서 배우나요?")
    print("답변:", result["answer"])
    print("출처:", [source["section"] for source in result["sources"]])
