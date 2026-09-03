"""Reusable prompt templates."""

from langchain_core.prompts import PromptTemplate


TRANSCRIPT_QA_PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer only from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""".strip(),
    input_variables=["context", "question"],
)
