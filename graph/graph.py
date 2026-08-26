from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import retrieve, grade_documents, generate, web_search
from graph.state import GraphState

load_dotenv(override=True)

def decide_to_generate(state: GraphState) -> bool:
    print("--- ASSESS GRADED DOCUMENTS ---")

    if state["web_search"]:
        print("--- DECIDE TO GENERATE: Web search required ---")
        return WEBSEARCH
    else:
        print("--- DECIDE TO GENERATE: No web search required ---")
        return GENERATE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,
                               decide_to_generate,
                               {WEBSEARCH: WEBSEARCH,
                                GENERATE: GENERATE},
                            )
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

print("=== workflow defined ===")

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")