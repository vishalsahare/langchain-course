import os
import pytest

from dotenv import load_dotenv
load_dotenv(override=True)

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY is required for this integration test", allow_module_level=True)

from pprint import pprint
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import router
from graph.chains.generation import generation_chain
from ingestion import retriever


@pytest.mark.integration
def test_retrieval_grader_answer_yes():

    question = "What are generative agents?"
    docs = retriever.invoke(question)
    assert docs, "The Chroma collection returned no documents"

    doc_txt = docs[0].page_content
    print(f"doc_txt: {doc_txt}")
    result = retrieval_grader.invoke({"document": doc_txt, "question": question})
    assert result.binary_score == "yes"


@pytest.mark.integration
def test_retrieval_grader_answer_no():

    question = "agent memory"
    docs = retriever.invoke(question)
    assert docs, "The Chroma collection returned no documents"

    doc_txt = docs[1].page_content
    result = retrieval_grader.invoke(
        {"document": doc_txt, "question": "How to make pizza"}
    )
    assert result.binary_score == "no"

@pytest.mark.integration
def test_generation_chain():
    question = "What are generative agents?"
    docs = retriever.invoke(question)
    result = generation_chain.invoke({"question": question, "context": docs})
    pprint(result)


@pytest.mark.integration
def test_hallucination_grader_yes():
    question = "What are generative agents?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    result = hallucination_grader.invoke({"documents": docs, "generation": generation})
    assert result.binary_score == "yes"


def test_hallucination_grader_no():
    question = "agent memory"
    docs = retriever.invoke(question)
    
    result = hallucination_grader.invoke({"documents": docs, "generation": "In order to make pizza we need to first start with the dough",})
    assert result.binary_score == "no"

def test_router_to_vectorstore() -> None:
    question = "What are generative agents?"
    result = router.invoke({"question": question})
    assert result.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "How to make pizza"
    result = router.invoke({"question": question})
    assert result.datasource == "websearch"