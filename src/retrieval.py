"""Retriever configuration."""

from typing import Literal

from langchain_community.vectorstores import FAISS


RetrievalMode = Literal["similarity", "mmr"]


def create_retriever(
    vector_store: FAISS,
    search_type: RetrievalMode = "similarity",
    k: int = 4,
    mmr_fetch_k: int = 12,
):
    """Create a similarity or MMR retriever for transcript chunks."""
    if search_type not in {"similarity", "mmr"}:
        raise ValueError("search_type must be 'similarity' or 'mmr'.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if search_type == "mmr" and mmr_fetch_k < k:
        raise ValueError("mmr_fetch_k must be greater than or equal to k.")

    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = mmr_fetch_k

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
