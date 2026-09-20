"""AI 교육과정 RAG 챗봇 Streamlit UI."""

import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI 교육과정 RAG 챗봇", page_icon="🎓")
st.title("🎓 AI 교육과정 RAG 챗봇")
st.caption("myclass.txt에 수록된 교육 단계, 시간, 학습 내용에 답합니다.")

with st.sidebar:
    st.header("설정")
    api_url = st.text_input("FastAPI 주소", value=DEFAULT_API_URL).rstrip("/")
    if st.button("대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("예시 질문")
    st.markdown(
        "- 전체 과정은 몇 시간인가요?\n"
        "- RAG는 어느 단계에서 배우나요?\n"
        "- STEP 6의 학습 내용은 무엇인가요?"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    with st.expander("참고한 문서"):
        for source in sources:
            st.markdown(f"**{source['section']} · {source['source']}**")
            st.text(source["content"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

if question := st.chat_input("교육과정에 관해 질문해 주세요"):
    prior_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-10:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("관련 교육 내용을 찾고 있습니다..."):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"question": question, "history": prior_history},
                    timeout=90,
                )
                response.raise_for_status()
                result = response.json()
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)
                show_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except requests.exceptions.ConnectionError:
                st.error("FastAPI 서버에 연결할 수 없습니다. 백엔드를 먼저 실행해 주세요.")
            except requests.exceptions.Timeout:
                st.error("답변 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.")
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = response.json().get("detail", str(exc))
                except ValueError:
                    detail = str(exc)
                st.error(f"API 오류: {detail}")
            except (KeyError, ValueError) as exc:
                st.error(f"API 응답 형식이 올바르지 않습니다: {exc}")
