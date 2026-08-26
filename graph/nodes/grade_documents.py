
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("---CHECK RELEVANCE OF DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        grade = score.binary_score
        if grade.lower() == "yes":
            print("--- GRADE: Document Relevant ---")
            filtered_docs.append(d)
        else:
            print("--- GRADE: Document Not Relevant ---")
            web_search = True

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
