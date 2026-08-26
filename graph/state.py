from typing import List, TypedDict


class GraphState(TypedDict):
    """State passed between nodes in the retrieval graph."""

    question: str
    generation: str
    web_search: bool
    documents: List[str]
