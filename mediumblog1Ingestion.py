import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import truststore

truststore.inject_into_ssl()

load_dotenv(override=True)

if __name__ == '__main__':
    print("Ingesting...")

    loader = UnstructuredLoader("mediumblog1.txt")
    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    # splitting
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} documents")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    PineconeVectorStore.from_documents(docs, embeddings, index_name=os.environ['INDEX_NAME'])

    print("finished ingesting")