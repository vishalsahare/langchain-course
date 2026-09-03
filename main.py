import truststore

truststore.inject_into_ssl()


from dotenv import load_dotenv

load_dotenv(override=True)

from graph.graph import app

def main():
    print("Hello Agentic RAG!")
    # print(app.invoke(input={"question": "What are generative agents?"}))
    print(app.invoke(input={"question": "How to make pizza?"}))

if __name__ == "__main__":
    main()
