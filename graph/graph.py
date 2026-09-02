from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
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

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> bool:

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("--- GRADE: Generation grounded in documents ---")
        print("--- GRADE GENERATION vs QUESTION")
        score = answer_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        if answer_grade := score.binary_score:
            print("--- GRADE: Generation answers question ---")
            return "useful"
        else:
            print("--- GRADE: Generation does not answer question ---")
            return "not useful"
    else:
        print("--- GRADE: Generation not grounded in documents ---")
        return "not supported"

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,
                               decide_to_generate,
                               {
                                   WEBSEARCH: WEBSEARCH,
                                   GENERATE: GENERATE
                            },
                            )

workflow.add_conditional_edges(GENERATE,
                               grade_generation_grounded_in_documents_and_question,
                               {
                                    "useful": END,
                                    "not useful": WEBSEARCH,
                                    "not supported": WEBSEARCH
                               },
                            )
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

print("=== workflow defined ===")

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")