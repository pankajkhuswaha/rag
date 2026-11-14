import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from agent.config import INITIAL_FETCH_K, RERANKER_MODEL, FINAL_TOP_K, SCORE_THRESHOLD


class RerankedRetriever:
    """Two-stage retriever: ChromaDB similarity search + CrossEncoder reranking with score thresholding."""

    def __init__(
        self,
        vectorstore: Chroma,
        initial_k: int = INITIAL_FETCH_K,
        top_k: int = FINAL_TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.vectorstore = vectorstore
        self.initial_k = initial_k
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.initial_k}
        )

        # Attempt local load first to avoid network latency/timeouts
        try:
            self.reranker = CrossEncoder(reranker_model, local_files_only=True)
        except Exception:
            self.reranker = CrossEncoder(reranker_model)

    def set_parameters(
        self,
        initial_k: Optional[int] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        """Allows dynamically updating retrieval depth and relevance threshold parameters."""
        if initial_k is not None:
            self.initial_k = initial_k
            self.base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.initial_k}
            )
        if top_k is not None:
            self.top_k = top_k
        if score_threshold is not None:
            self.score_threshold = score_threshold

    def retrieve(self, question: str) -> List[Document]:
        """Fetches candidates, scores them with CrossEncoder, and filters out chunks below threshold."""
        candidates = self.base_retriever.invoke(question)
        if not candidates:
            return []

        pairs = [[question, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        top_docs = []
        for score, doc in scored_docs[: self.top_k]:
            numeric_score = float(score)
            # Anti-hallucination filter: discard chunks that fail relevance cutoff
            if numeric_score >= self.score_threshold:
                doc.metadata["rerank_score"] = round(numeric_score, 4)
                top_docs.append(doc)

        return top_docs


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved Document objects into a clean text block for the system prompt."""
    if not docs:
        return "NO_RELEVANT_INFORMATION_FOUND"

    formatted_blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename") or doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        formatted_blocks.append(f"--- [Reference {i}: {source}{page_info}] ---\n{doc.page_content}")
    return "\n\n".join(formatted_blocks)
