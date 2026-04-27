from agent import LangGraphAgent
from config import mongo_client


def main():
    """LangGraph and MongoDB agent with tools and memory."""
    agent = LangGraphAgent()
    thread_id = input("Enter a session ID: ").strip()
    print("Ask me about movies! Type 'quit' to exit.")

    try:
        while True:
            user_query = input("\nYour question: ").strip()

            if user_query.lower() == "quit":
                break

            answer = agent.execute(user_query, thread_id)
            print(f"\nAnswer: {answer}")

    finally:
        mongo_client.close()


if __name__ == "__main__":
    main()
