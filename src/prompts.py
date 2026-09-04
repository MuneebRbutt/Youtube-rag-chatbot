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


CONVERSATION_REWRITE_PROMPT = PromptTemplate(
    template="""
Rewrite the latest user question as a standalone question for transcript search.
Use the conversation only to resolve references such as "it", "they", or
"that topic". Do not answer the question, add details, or mention the
conversation. If the question is already standalone, return it unchanged.

Conversation:
{history}

Latest question:
{question}
""".strip(),
    input_variables=["history", "question"],
)


TRANSCRIPT_MAP_SUMMARY_PROMPT = PromptTemplate(
    template="""
Summarize this section of a YouTube transcript. Preserve the important facts,
arguments, examples, and conclusions. Do not add information that is not in the
transcript.

Transcript section:
{chunk}
""".strip(),
    input_variables=["chunk"],
)


TRANSCRIPT_REDUCE_SUMMARY_PROMPT = PromptTemplate(
    template="""
Write a coherent full-video summary from these partial transcript summaries.
Cover the main ideas, supporting arguments, and final conclusions. Only use the
provided summaries; do not add outside information.

Partial summaries:
{summaries}
""".strip(),
    input_variables=["summaries"],
)
