from dotenv import load_dotenv
load_dotenv(override=True)
import os
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
    chunk_size = 250, chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings()

# Chroma.from_documents(
#     documents=doc_splits,
#     embedding=embeddings,
#     collection_name="rag-chroma",
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()