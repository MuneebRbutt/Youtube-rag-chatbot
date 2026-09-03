"""Command-line entry point for the YouTube transcript RAG application."""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import build_timestamp_aware_chunks
from src.ingestion import transcript_character_count, transcript_to_documents
from src.rag import answer_question
from src.retrieval import create_retriever
from src.utils import MAX_TRANSCRIPT_CHARACTERS, configure_logging, show_error
from src.vectorstore import (
    create_embeddings,
    create_vector_store,
    load_vector_store,
    save_vector_store,
)
from src.youtube import extract_video_id, fetch_english_transcript, transcript_error_message


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse the video URL, user question, and optional FAISS persistence path."""
    parser = argparse.ArgumentParser(description="Ask questions about a YouTube transcript.")
    parser.add_argument("url", help="YouTube watch, short, or youtu.be URL")
    parser.add_argument(
        "--question",
        default="Can you summarize the video?",
        help="Question to answer from the transcript",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Optional directory in which to save or load a local FAISS index",
    )
    return parser.parse_args()


def print_sources(sources) -> None:
    """Print timestamp ranges for the chunks used to answer the question."""
    if not sources:
        return
    print("\nSources:")
    for source in sources:
        print(
            f"- {source.metadata['video_id']} "
            f"({source.metadata['start_time']:.1f}s-{source.metadata['end_time']:.1f}s)"
        )


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
        video_id = extract_video_id(arguments.url)
    except ValueError as error:
        logger.warning("Invalid YouTube URL provided: %s", error)
        show_error(str(error), "Paste a valid YouTube watch, short, or youtu.be URL.")
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
    else:
        try:
            transcript = fetch_english_transcript(video_id)
            transcript_documents = transcript_to_documents(transcript, video_id)
        except Exception as error:
            logger.exception("Transcript retrieval failed for video ID %s", video_id)
            show_error(transcript_error_message(error), "Please try another video.")
            return

        if not transcript_documents:
            show_error(
                "The transcript is empty or contains no usable caption text.",
                "Please try another video.",
            )
            return

        if transcript_character_count(transcript_documents) > MAX_TRANSCRIPT_CHARACTERS:
            logger.warning("Transcript exceeds the configured size limit: %s", video_id)
            show_error(
                "This transcript is too long to process safely in one run.",
                "Please try a shorter video.",
            )
            return

        print(f"Loaded {len(transcript_documents)} transcript segments.")
        try:
            chunks = build_timestamp_aware_chunks(transcript_documents)
        except Exception:
            logger.exception("Transcript chunking failed for video ID %s", video_id)
            show_error(
                "The transcript could not be prepared for searching.",
                "Please try another video.",
            )
            return

        if not chunks:
            show_error(
                "The transcript could not be prepared for searching.",
                "Please try another video.",
            )
            return

        try:
            embeddings, vectors = create_embeddings(chunks)
        except Exception:
            logger.exception("Embedding creation failed for video ID %s", video_id)
            show_error(
                "OpenAI could not create embeddings for this transcript.",
                "Check your API key and account, then try again.",
            )
            return

        try:
            vector_store = create_vector_store(chunks, embeddings, vectors)
        except Exception:
            logger.exception("FAISS creation failed for video ID %s", video_id)
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
        result = answer_question(create_retriever(vector_store), arguments.question)
    except Exception:
        logger.exception("OpenAI answer generation failed for video ID %s", video_id)
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
