from typing import List, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str. Any]:
    print("---RETRIEVE---")
    question = state["question"]
    docs = retriever.invoke(question)
    return {"docs": docs, "question": question}