from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv(override=True)


llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [TavilySearch()]

agent = create_agent(llm, tools)


def main():
    print("Hello from langchain-course!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Seach for 3 job postings for AI Engineer using langchain in the bay area on linkedin and list their details")]}
    )
    print(response)


if __name__ == "__main__":
    main()
