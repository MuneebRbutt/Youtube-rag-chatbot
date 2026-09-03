"""Retriever configuration."""

from langchain_community.vectorstores import FAISS


def create_retriever(vector_store: FAISS, k: int = 4):
    """Create a similarity retriever for the most relevant transcript chunks."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
