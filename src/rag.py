"""RAG chain assembly and source-aware answers."""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.prompts import (
    CONVERSATION_REWRITE_PROMPT,
    TRANSCRIPT_MAP_SUMMARY_PROMPT,
    TRANSCRIPT_QA_PROMPT,
    TRANSCRIPT_REDUCE_SUMMARY_PROMPT,
)


SUMMARY_REDUCE_BATCH_CHARACTERS = 12_000
MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARACTERS = 6_000


@dataclass
class RAGAnswer:
    """A generated answer and the transcript chunks used as context."""

    answer: str
    sources: list[Document]


@dataclass
class ConversationTurn:
    """One user question and its assistant answer."""

    question: str
    answer: str


def format_documents(documents: list[Document]) -> str:
    """Join retrieved transcript chunks into prompt context."""
    return "\n\n".join(
        "[Video: "
        f"{document.metadata.get('video_title', document.metadata['video_id'])}; "
        f"starts at {format_timestamp(document.metadata['start_time'])}]\n"
        f"{document.page_content}"
        for document in documents
    )


def format_timestamp(seconds: float) -> str:
    """Format a video offset as a readable timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_source_citations(sources: list[Document]) -> list[str]:
    """Create de-duplicated Markdown links to retrieved video timestamps."""
    citations: list[str] = []
    seen: set[tuple[str, int]] = set()

    for source in sources:
        video_id = str(source.metadata["video_id"])
        video_title = str(source.metadata.get("video_title", video_id))
        start_seconds = max(0, int(float(source.metadata["start_time"])))
        key = (video_id, start_seconds)
        if key in seen:
            continue
        seen.add(key)

        timestamp = format_timestamp(start_seconds)
        url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
        citations.append(f"[{video_title} at {timestamp}]({url})")

    return citations


def create_answer_chain():
    """Build the LCEL prompt, model, and output-parser chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return TRANSCRIPT_QA_PROMPT | llm | StrOutputParser()


def format_conversation_history(
    history: list[ConversationTurn],
    max_characters: int = MAX_HISTORY_CHARACTERS,
) -> str:
    """Format recent turns solely for standalone-question rewriting."""
    if not history:
        return "No previous conversation."

    selected_turns: list[str] = []
    remaining_characters = max_characters
    for turn in reversed(history):
        formatted_turn = f"User: {turn.question}\nAssistant: {turn.answer}"
        if len(formatted_turn) > remaining_characters:
            formatted_turn = formatted_turn[:remaining_characters]
        selected_turns.append(formatted_turn)
        remaining_characters -= len(formatted_turn)
        if remaining_characters <= 0:
            break

    return "\n\n".join(reversed(selected_turns))


def create_question_rewriter():
    """Build the LCEL chain that converts follow-ups into retrieval queries."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return CONVERSATION_REWRITE_PROMPT | llm | StrOutputParser()


def rewrite_question(
    question: str,
    history: list[ConversationTurn],
    max_history_turns: int = MAX_HISTORY_TURNS,
    max_history_characters: int = MAX_HISTORY_CHARACTERS,
) -> str:
    """Resolve follow-up references without sending chat history to retrieval."""
    if not history:
        return question

    chain = create_question_rewriter()
    rewritten = chain.invoke(
        {
            "history": format_conversation_history(
                history[-max_history_turns:],
                max_characters=max_history_characters,
            ),
            "question": question,
        }
    ).strip()
    return rewritten or question


def create_summary_chains():
    """Build LCEL chains for map and reduce stages of a full-video summary."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    return (
        TRANSCRIPT_MAP_SUMMARY_PROMPT | llm | parser,
        TRANSCRIPT_REDUCE_SUMMARY_PROMPT | llm | parser,
    )


def answer_question(retriever, question: str) -> RAGAnswer:
    """Retrieve context, invoke the RAG chain, and return answer with sources."""
    sources = retriever.invoke(question)
    chain = create_answer_chain()
    answer = chain.invoke({"context": format_documents(sources), "question": question})
    return RAGAnswer(answer=answer, sources=sources)


def batch_summaries(
    summaries: list[str], max_characters: int = SUMMARY_REDUCE_BATCH_CHARACTERS
) -> list[str]:
    """Group partial summaries into context-safe batches for reduction."""
    batches: list[str] = []
    current_batch: list[str] = []
    current_size = 0

    for summary in summaries:
        summary_size = len(summary)
        if (
            len(current_batch) >= 2
            and current_size + summary_size > max_characters
        ):
            batches.append("\n\n".join(current_batch))
            current_batch = []
            current_size = 0
        current_batch.append(summary)
        current_size += summary_size

    if current_batch:
        batches.append("\n\n".join(current_batch))
    return batches


def select_summary_sources(chunks: list[Document], limit: int = 10) -> list[Document]:
    """Select evenly distributed timestamp links without listing every chunk."""
    if limit < 1:
        return []
    if len(chunks) <= limit:
        return chunks
    if limit == 1:
        return [chunks[0]]
    indexes = {
        round(position * (len(chunks) - 1) / (limit - 1))
        for position in range(limit)
    }
    return [chunk for index, chunk in enumerate(chunks) if index in indexes]


def summarize_video(chunks: list[Document]) -> RAGAnswer:
    """Summarize every transcript chunk through hierarchical map-reduce steps."""
    if not chunks:
        raise ValueError("Cannot summarize an empty transcript.")

    map_chain, reduce_chain = create_summary_chains()
    summaries = [map_chain.invoke({"chunk": chunk.page_content}) for chunk in chunks]

    while len(summaries) > 1:
        summaries = [
            reduce_chain.invoke({"summaries": batch})
            for batch in batch_summaries(summaries)
        ]

    return RAGAnswer(answer=summaries[0], sources=select_summary_sources(chunks))
