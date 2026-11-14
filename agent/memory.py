# agent/memory.py
from langgraph.checkpoint.memory import InMemorySaver

# Global checkpointer storing conversation state per thread_id
# (Can be swapped for SqliteSaver or PostgresSaver later without touching chain.py)
checkpointer = InMemorySaver()


def clear_session_history(thread_id: str) -> None:
    """Clears all saved checkpoints/messages for a specific thread_id."""
    checkpointer.delete_thread(thread_id)