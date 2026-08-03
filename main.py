from typing import List, Dict, Any
from backend.core import run_llm
import streamlit as st

def _format_sources(context_docs: List[Any]):
    return [
        str(meta.get("source", "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="LangChain Documentation Chatbot", layout="centered")
st.title("LangChain Documentation Chatbot")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop('messages', None)
        st.rerun()

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            'role': 'assistant',
            'content': 'How can I help you?',
            'sources': [],
        }
    ]

def main():
    query = "What are deep agents?"
#     result = run_llm(query)
#     print(result)

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('sources'):
            with st.expander("Sources"):
                for s in msg['sources']:
                    st.markdown(f" - {s}")

prompt = st.chat_input("Ask a question about Langchain...")
if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(Sorry, I don't know.)"
                sources = _format_sources(result.get("artifacts", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f" - {s}")

            st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': sources})
            # st.rerun()

        except Exception as e:
            st.error("Failed to generate a response")
            st.exception(e)

