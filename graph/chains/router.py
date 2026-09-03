from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Given a user question choose to route it to vectorstore or websearch"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

router = router_prompt | structured_llm_router