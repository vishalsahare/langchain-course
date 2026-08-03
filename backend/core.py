import os
from typing import List, Dict, Any
from logger import log_warning, log_info, log_error, log_success, log_header
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import truststore

truststore.inject_into_ssl()

load_dotenv(override=True)

# Initialize embedding models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = PineconeVectorStore(embedding=embeddings, index_name="langchain-doc-index")

# Initialize chat model
model = init_chat_model("gpt-5.2", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant context from documentation for a given query."""

    # Retrieve top 4 relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
    if retrieved_docs is None or len(retrieved_docs) == 0:
        log_warning(f"No relevant documents found for query: {query}")
        return "No relevant context found."

    # Serialize documents for model
    serialized = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in retrieved_docs])

    return serialized, retrieved_docs

def run_llm(query: str):
    """
    Run the LLM with the given query.
    
    Args:
        query (str): The user's query.
    
    Returns:
        Dictionary containing:
            - answer (str): The model's response.
            - artifacts (list): List of artifacts (e.g., retrieved documents).
    """

    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # create agent for specific model having system prompt and tools
    agent = create_agent(model=model, system_prompt=system_prompt, tools=[retrieve_context])

    # Build message list
    messages = [
        {"role": "user", "content": query}
    ]

    # Run the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response['messages'][-1].content

    # Extract context documents from ToolMessage
    context_docs = []
    for message in response['messages']:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # artifact should contain the list of Documents objects
            context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "artifacts": context_docs
    }

if __name__ == "__main__":
    query = "What are deep agents?"
    result = run_llm(query)
    print(result)
