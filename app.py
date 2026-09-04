"""Command-line entry point for the YouTube transcript RAG application."""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import build_timestamp_aware_chunks
from src.ingestion import transcript_character_count, transcript_to_documents
from src.rag import (
    MAX_HISTORY_TURNS,
    ConversationTurn,
    answer_question,
    format_source_citations,
    rewrite_question,
    summarize_video,
)
from src.retrieval import create_retriever
from src.utils import MAX_TRANSCRIPT_CHARACTERS, configure_logging, show_error
from src.vectorstore import (
    add_documents,
    create_embeddings,
    create_vector_store,
    load_vector_store,
    save_vector_store,
)
from src.youtube import (
    extract_video_id,
    fetch_english_transcript,
    fetch_video_title,
    transcript_error_message,
)


logger = logging.getLogger(__name__)


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def parse_arguments() -> argparse.Namespace:
    """Parse the video URL, operating mode, and optional FAISS persistence path."""
    parser = argparse.ArgumentParser(description="Ask questions about a YouTube transcript.")
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more YouTube watch, short, or youtu.be URLs",
    )
    parser.add_argument(
        "--mode",
        choices=("question", "summary"),
        default="question",
        help="Use retrieval for a question or map-reduce for a full-video summary",
    )
    parser.add_argument(
        "--question",
        default="What are the main points of the video?",
        help="Question to answer in question mode",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Optional directory in which to save or load a local FAISS index",
    )
    parser.add_argument(
        "--add-to-index",
        action="store_true",
        help="Add the supplied videos to an existing --index-dir knowledge base",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("similarity", "mmr"),
        default="similarity",
        help="Retrieval strategy for question mode (default: similarity)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=positive_integer,
        default=4,
        help="Number of transcript chunks to provide to question answering",
    )
    parser.add_argument(
        "--mmr-fetch-k",
        type=positive_integer,
        default=12,
        help="Candidate chunks considered by MMR before selecting retrieval-k",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive question-answering session with follow-up support",
    )
    return parser.parse_args()


def print_sources(sources) -> None:
    """Print direct links to the timestamps used to answer the question."""
    citations = format_source_citations(sources)
    if not citations:
        return
    print("\nSources:")
    for citation in citations:
        print(f"- {citation}")


def fetch_and_chunk_video(video_id: str):
    """Fetch, validate, and timestamp-chunk one video for a knowledge base."""
    try:
        video_title = fetch_video_title(video_id)
    except Exception:
        logger.exception("Video title retrieval failed for video ID %s", video_id)
        show_error(
            "The video title could not be retrieved.",
            "Please try another public YouTube video.",
        )
        return None

    try:
        transcript = fetch_english_transcript(video_id)
        transcript_documents = transcript_to_documents(transcript, video_id, video_title)
    except Exception as error:
        logger.exception("Transcript retrieval failed for video ID %s", video_id)
        show_error(transcript_error_message(error), "Please try another video.")
        return None

    if not transcript_documents:
        show_error(
            "The transcript is empty or contains no usable caption text.",
            "Please try another video.",
        )
        return None

    if transcript_character_count(transcript_documents) > MAX_TRANSCRIPT_CHARACTERS:
        logger.warning("Transcript exceeds the configured size limit: %s", video_id)
        show_error(
            "This transcript is too long to process safely in one run.",
            "Please try a shorter video.",
        )
        return None

    print(f'Loaded {len(transcript_documents)} transcript segments from "{video_title}".')
    try:
        chunks = build_timestamp_aware_chunks(transcript_documents)
    except Exception:
        logger.exception("Transcript chunking failed for video ID %s", video_id)
        show_error(
            "The transcript could not be prepared for searching.",
            "Please try another video.",
        )
        return None

    if not chunks:
        show_error(
            "The transcript could not be prepared for searching.",
            "Please try another video.",
        )
        return None
    return chunks


def fetch_and_chunk_videos(video_ids: list[str]):
    """Build timestamp-aware chunks for every requested video."""
    all_chunks = []
    for video_id in video_ids:
        chunks = fetch_and_chunk_video(video_id)
        if chunks is None:
            return None
        all_chunks.extend(chunks)
    return all_chunks


def run_chat(retriever) -> None:
    """Answer follow-up questions by rewriting them before retrieval."""
    history: list[ConversationTurn] = []
    print("Chat mode. Ask a question, or press Enter or type 'exit' to finish.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print()
            return

        if not question or question.lower() in {"exit", "quit"}:
            return

        try:
            standalone_question = rewrite_question(question, history)
        except Exception:
            logger.exception("Conversation question rewriting failed")
            show_error(
                "OpenAI could not interpret this follow-up question.",
                "Please rephrase it with more detail.",
            )
            continue

        if standalone_question != question:
            print(f"Searching for: {standalone_question}")

        try:
            result = answer_question(retriever, standalone_question)
        except Exception:
            logger.exception("Chat answer generation failed")
            show_error(
                "OpenAI could not generate an answer for this question.",
                "Check your API key and account, then try again.",
            )
            continue

        print(f"Assistant: {result.answer}")
        print_sources(result.sources)
        history.append(ConversationTurn(question=question, answer=result.answer))
        history = history[-MAX_HISTORY_TURNS:]


def main() -> None:
    """Run the transcript ingestion, retrieval, and answer-generation pipeline."""
    configure_logging()
    arguments = parse_arguments()
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        show_error(
            "OPENAI_API_KEY is missing.",
            "Add it to .env and run the application again.",
        )
        return

    try:
        video_ids = list(dict.fromkeys(extract_video_id(url) for url in arguments.urls))
    except ValueError as error:
        logger.warning("Invalid YouTube URL provided: %s", error)
        show_error(str(error), "Paste a valid YouTube watch, short, or youtu.be URL.")
        return

    if arguments.mode == "summary" and arguments.index_dir:
        show_error(
            "--index-dir is only available in question mode.",
            "Run the full summary without --index-dir.",
        )
        return

    if arguments.mode == "summary" and len(video_ids) > 1:
        show_error(
            "Full-video summary supports one YouTube URL at a time.",
            "Use question mode to search across multiple videos.",
        )
        return

    if arguments.mode == "summary" and arguments.chat:
        show_error(
            "--chat is only available in question mode.",
            "Run the full summary without --chat.",
        )
        return

    if (
        arguments.retrieval_mode == "mmr"
        and arguments.mmr_fetch_k < arguments.retrieval_k
    ):
        show_error(
            "--mmr-fetch-k must be greater than or equal to --retrieval-k.",
            "Use a larger MMR candidate pool or reduce --retrieval-k.",
        )
        return

    if arguments.mode == "summary":
        chunks = fetch_and_chunk_video(video_ids[0])
        if chunks is None:
            return
        try:
            result = summarize_video(chunks)
        except Exception:
            logger.exception("Full-video summarization failed for video ID %s", video_ids[0])
            show_error(
                "OpenAI could not summarize this full video.",
                "Check your API key and account, then try again.",
            )
            return

        print(result.answer)
        print_sources(result.sources)
        return

    if arguments.index_dir and arguments.index_dir.exists():
        try:
            vector_store = load_vector_store(arguments.index_dir)
        except Exception:
            logger.exception("FAISS index load failed from %s", arguments.index_dir)
            show_error(
                "The saved transcript search index could not be loaded.",
                "Delete the index directory or try again without --index-dir.",
            )
            return
        if arguments.add_to_index:
            chunks = fetch_and_chunk_videos(video_ids)
            if chunks is None:
                return
            try:
                add_documents(vector_store, chunks)
                save_vector_store(vector_store, arguments.index_dir)
            except Exception:
                logger.exception("Adding videos to FAISS index failed")
                show_error(
                    "The videos could not be added to the search index.",
                    "Check your API key and try again.",
                )
                return
    else:
        chunks = fetch_and_chunk_videos(video_ids)
        if chunks is None:
            return

        try:
            embeddings, vectors = create_embeddings(chunks)
        except Exception:
            logger.exception("Embedding creation failed for the video knowledge base")
            show_error(
                "OpenAI could not create embeddings for this transcript.",
                "Check your API key and account, then try again.",
            )
            return

        try:
            vector_store = create_vector_store(chunks, embeddings, vectors)
        except Exception:
            logger.exception("FAISS creation failed for the video knowledge base")
            show_error(
                "The transcript search index could not be created.",
                "Please try again or use a shorter video.",
            )
            return

        if arguments.index_dir:
            try:
                save_vector_store(vector_store, arguments.index_dir)
            except Exception:
                logger.exception("FAISS index save failed to %s", arguments.index_dir)
                show_error(
                    "The transcript was processed, but its search index could not be saved.",
                    "Try again without --index-dir or choose a writable location.",
                )
                return

    try:
        retriever = create_retriever(
            vector_store,
            search_type=arguments.retrieval_mode,
            k=arguments.retrieval_k,
            mmr_fetch_k=arguments.mmr_fetch_k,
        )
    except Exception:
        logger.exception("Retriever creation failed")
        show_error(
            "The transcript search could not be configured.",
            "Check the retrieval options and try again.",
        )
        return

    print(f"Retrieval mode: {arguments.retrieval_mode}")
    if arguments.chat:
        run_chat(retriever)
        return

    try:
        result = answer_question(retriever, arguments.question)
    except Exception:
        logger.exception("OpenAI answer generation failed for the video knowledge base")
        show_error(
            "OpenAI could not generate an answer for this video.",
            "Check your API key and account, then try again.",
        )
        return

    print(result.answer)
    print_sources(result.sources)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe application was cancelled.")
    except Exception:
        logger.exception("Unexpected application failure")
        show_error(
            "The application could not complete the request.",
            "Please try again. Technical details were saved to rag_app.log.",
        )
