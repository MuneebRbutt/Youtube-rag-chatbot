"""RAG chain assembly and source-aware answers."""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.prompts import TRANSCRIPT_QA_PROMPT


@dataclass
class RAGAnswer:
    """A generated answer and the transcript chunks used as context."""

    answer: str
    sources: list[Document]


def format_documents(documents: list[Document]) -> str:
    """Join retrieved transcript chunks into prompt context."""
    return "\n\n".join(document.page_content for document in documents)


def create_answer_chain():
    """Build the LCEL prompt, model, and output-parser chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return TRANSCRIPT_QA_PROMPT | llm | StrOutputParser()


def answer_question(retriever, question: str) -> RAGAnswer:
    """Retrieve context, invoke the RAG chain, and return answer with sources."""
    sources = retriever.invoke(question)
    chain = create_answer_chain()
    answer = chain.invoke({"context": format_documents(sources), "question": question})
    return RAGAnswer(answer=answer, sources=sources)
