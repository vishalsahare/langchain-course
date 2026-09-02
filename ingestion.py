from dotenv import load_dotenv

load_dotenv(override=True)
import os
from pathlib import Path

os.environ.setdefault("USER_AGENT", "langchain-course/0.1")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


import truststore

truststore.inject_into_ssl()


URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

persist_directory = Path(__file__).resolve().parent / ".chroma"


def _build_retriever():
    """Create the vector-store retriever, loading source pages only if needed."""
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="rag-chroma",
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )

    if not vector_store.get(limit=1)["ids"]:
        docs = [WebBaseLoader(url).load() for url in URLS]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        vector_store.add_documents(doc_splits)

    return vector_store.as_retriever()


class _LazyRetriever:
    """Defer embeddings, disk access, and web loading until retrieval is used."""

    def __init__(self):
        self._retriever = None

    def _get_retriever(self):
        if self._retriever is None:
            self._retriever = _build_retriever()
        return self._retriever

    def invoke(self, *args, **kwargs):
        return self._get_retriever().invoke(*args, **kwargs)


retriever = _LazyRetriever()
