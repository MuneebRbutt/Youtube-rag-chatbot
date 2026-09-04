"""FAISS index construction and local persistence."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def create_embeddings(chunks: list[Document]) -> tuple[OpenAIEmbeddings, list[list[float]]]:
    """Create OpenAI embeddings for transcript chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    return embeddings, vectors


def create_vector_store(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    vectors: list[list[float]],
) -> FAISS:
    """Create an in-memory FAISS index from already-created embeddings."""
    return FAISS.from_embeddings(
        text_embeddings=[
            (chunk.page_content, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )


def add_documents(vector_store: FAISS, chunks: list[Document]) -> None:
    """Embed and add chunks from additional videos to an existing FAISS index."""
    vector_store.add_documents(chunks)


def save_vector_store(vector_store: FAISS, index_directory: str | Path) -> None:
    """Persist a FAISS index for a later local load."""
    vector_store.save_local(str(index_directory))


def load_vector_store(index_directory: str | Path) -> FAISS:
    """Load an index created by this application from a trusted local path."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        str(index_directory),
        embeddings,
        allow_dangerous_deserialization=True,
    )
