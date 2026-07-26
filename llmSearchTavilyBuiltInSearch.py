from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv(override=True)

class Source(BaseModel):
    """Schema of the source used by Agents"""

    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Schema of the agent response with answer and sources"""

    answer: str = Field(description="Answer to the user's query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer the query")

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Seach for 3 job postings for AI Engineer using langchain in the bay area on linkedin and list their details")]}
    )
    print(response)


if __name__ == "__main__":
    main()
