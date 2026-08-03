import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import log_header, log_info, log_success, log_error, Colors

load_dotenv(override=True)
import truststore

truststore.inject_into_ssl()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10
)

#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = PineconeVectorStore(embedding=embeddings, index_name="langchain-doc-index")
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")
    log_info(f"  VectorStore Indexing: Preparing to index {len(documents)} documents in batches of {batch_size}")

    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"  VectorStore Indexing: Splitting into {len(batches)} batches of {batch_size} documents each")

    # processing all batches sucessfully
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"  VectorStore Indexing: Indexed batch {batch_num} of {len(batch)} documents")
        except Exception as e:
            log_error(f"  VectorStore Indexing: Error indexing batch {batch_num}: {e}")
            return False
        return True
    
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"  VectorStore Indexing: Successfully indexed {len(documents)} documents")
    else:
        log_error(f"  VectorStore Indexing: Failed to index {len(batches) - successful} batches")

async def main():
    """Main async function to orchestrate the entire process"""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("  TavalyCrawl: Starting to Crawl documentation from https://python.langchain.com/")

    # Crawl the documentation
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res['results']]
    log_success(f"  TavalyCrawl: Crawl complete. Found {len(all_docs)} documents")

    # split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info("  TextSplitter: Porcessing {len(all_docs)} to split documents with 4000 chunks and 200 overlaps")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"  TextSplitter: create {len(splitted_docs)} chunks from {len(all_docs)} documents")

    await index_documents_async(splitted_docs, batch_size=200)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())
    print("Ingestion complete")
