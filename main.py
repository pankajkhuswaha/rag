# main.py
from langchain_core.messages import HumanMessage
from agent.ingestion import get_vectorstore
from agent.retriever import RerankedRetriever
from agent.chain import build_rag_chain
from agent.memory import clear_session_history
from agent import config


def run_cli():
    vectorstore = get_vectorstore()
    retriever = RerankedRetriever(vectorstore)

    provider = config.LLM_PROVIDER
    model = config.GROQ_MODEL if provider == "groq" else config.LLM_MODEL
    rag_app = build_rag_chain(
        retriever=retriever,
        provider=provider,
        model_name=model,
    )

    thread_id = "local_cli_student"
    run_config = {"configurable": {"thread_id": thread_id}}

    engine_name = f"Groq Cloud ({model})" if provider == "groq" else f"Local Ollama ({model})"
    print(f"\nAmi (Amity University Admissions Counselor - {engine_name}) is ready!")
    print("Type 'exit' to quit or 'clear' to reset conversation memory.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "clear":
            clear_session_history(thread_id)
            print("Conversation memory cleared.\n")
            continue
        if not query:
            continue

        print("\nAmi is thinking...")
        result = rag_app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=run_config
        )
        answer = result["messages"][-1].content
        print(f"\nAmi:\n{answer}\n" + "-" * 50)


if __name__ == "__main__":
    run_cli()
