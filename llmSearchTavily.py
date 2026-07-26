from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv(override=True)
tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """Search the web for the given query and return the result."""
    print(f"Searching for {query}")
    # return "Tokyo weather is sunny"
    return tavily.search(query)


llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [search]

agent = create_agent(llm, tools)


def main():
    print("Hello from langchain-course!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo today?")]}
    )
    print(response)


if __name__ == "__main__":
    main()
