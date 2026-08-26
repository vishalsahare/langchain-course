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


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings()
persist_directory = Path(__file__).resolve().parent / ".chroma"

vector_store = Chroma(
    collection_name="rag-chroma",
    persist_directory=str(persist_directory),
    embedding_function=embeddings,
)

if not vector_store.get(limit=1)["ids"]:
    vector_store.add_documents(doc_splits)

retriever = vector_store.as_retriever()
