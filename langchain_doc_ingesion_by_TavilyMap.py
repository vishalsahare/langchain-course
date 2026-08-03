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

from logger import log_header, log_info, log_success, log_error, Colors, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embeddings-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10
)

#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = PineconeVectorStore(embedding=embeddings, index_name="langchain-doc-index")
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i:i + chunk_size])
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:

    try:
        log_info(f"  TavilyExtract: Processing batch {batch_num} of {len(urls)} URLs")
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"  TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents"
        )
        return docs
    except Exception as e:
        log_error(f"  TavilyExtract: Error processing batch {batch_num}: {e}")
        return []

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACT PHASE")
    log_info(f"  TavilyExtract: Starting to extract content from {len(url_batches)} batches of URLs")
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
        else:
            for extracted_page in result['results']:
                document = Document(
                    page_content = extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
            all_pages.append(document)

    log_success(f"  TavilyExtract: Completed all batches. Extracted {len(all_pages)} documents")

    if failed_batches > 1:
        log_warning(f"  TavilyExtract: {failed_batches} batches failed to extract")

    return all_pages

async def main():
    """Main async function to orchestrate the entire process"""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("  TavilyMap: Starting to Crawl documentation from https://python.langchain.com/")

    # Crawl the documentation
    site_map = tavily_map.invoke("https://python.langchain.com/")

    log_success(f"  TavilyMap: Successfully Mapped {len(site_map['results'])} URLs from documentation site")

    urls_batches = chunk_urls(site_map['results'], chunk_size=20)
    log_info(
        f"  URL Processing: Split {len(site_map['results'])} URLs into {len(urls_batches)} batches"
    )

    # Extract document from URLs
    all_docs = await async_extract(urls_batches)


if __name__ == "__main__":
    asyncio.run(main())
    print("Ingestion complete")
