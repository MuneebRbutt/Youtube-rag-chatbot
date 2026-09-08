"""Streamlit frontend for the YouTube Knowledge Assistant."""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from src.chunking import build_timestamp_aware_chunks
from src.ingestion import transcript_character_count, transcript_to_documents
from src.rag import (
    ConversationTurn,
    answer_question,
    format_source_citations,
    rewrite_question,
    summarize_video,
)
from src.retrieval import create_retriever
from src.utils import MAX_TRANSCRIPT_CHARACTERS, configure_logging
from src.vectorstore import create_embeddings, create_vector_store
from src.youtube import (
    extract_video_id,
    fetch_english_transcript,
    fetch_video_title,
    transcript_error_message,
)


logger = logging.getLogger(__name__)


def initialize_session() -> None:
    """Create the per-browser-session state used by the interface."""
    defaults = {
        "chunks": [],
        "vector_store": None,
        "video_titles": [],
        "chat_history": [],
        "messages": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def parse_urls(raw_urls: str) -> list[str]:
    """Validate and de-duplicate one YouTube URL per input line."""
    urls = [line.strip() for line in raw_urls.splitlines() if line.strip()]
    if not urls:
        raise ValueError("Paste at least one YouTube URL.")

    video_ids = [extract_video_id(url) for url in urls]
    return list(dict.fromkeys(video_ids))


def process_videos(video_ids: list[str]) -> tuple[list, object, list[str]]:
    """Fetch, enrich, chunk, embed, and index every selected video."""
    all_chunks = []
    video_titles: list[str] = []
    progress = st.progress(0, text="Preparing videos...")

    for index, video_id in enumerate(video_ids, start=1):
        progress.progress(
            (index - 1) / len(video_ids),
            text=f"Fetching video {index} of {len(video_ids)}...",
        )
        try:
            title = fetch_video_title(video_id)
        except Exception as error:
            logger.exception("Video title retrieval failed for %s", video_id)
            raise RuntimeError(
                "The video title could not be retrieved. Please try another public YouTube video."
            ) from error

        try:
            transcript = fetch_english_transcript(video_id)
            documents = transcript_to_documents(transcript, video_id, title)
        except Exception as error:
            logger.exception("Transcript retrieval failed for %s", video_id)
            raise RuntimeError(
                f"{transcript_error_message(error)} Please try another video."
            ) from error

        if not documents:
            raise RuntimeError(
                "The transcript is empty or contains no usable caption text. "
                "Please try another video."
            )
        if transcript_character_count(documents) > MAX_TRANSCRIPT_CHARACTERS:
            raise RuntimeError(
                "This transcript is too long to process safely in one run. "
                "Please try a shorter video."
            )

        try:
            chunks = build_timestamp_aware_chunks(documents)
        except Exception as error:
            logger.exception("Transcript chunking failed for %s", video_id)
            raise RuntimeError(
                "The transcript could not be prepared for searching. Please try another video."
            ) from error

        if not chunks:
            raise RuntimeError(
                "The transcript could not be prepared for searching. Please try another video."
            )
        all_chunks.extend(chunks)
        video_titles.append(title)

    progress.progress(0.8, text="Creating the searchable knowledge base...")
    try:
        embeddings, vectors = create_embeddings(all_chunks)
    except Exception as error:
        logger.exception("Embedding creation failed in Streamlit frontend")
        raise RuntimeError(
            "OpenAI could not create embeddings for these videos. "
            "Check your API key and account, then try again."
        ) from error

    try:
        vector_store = create_vector_store(all_chunks, embeddings, vectors)
    except Exception as error:
        logger.exception("FAISS creation failed in Streamlit frontend")
        raise RuntimeError(
            "The transcript search index could not be created. Please try again."
        ) from error

    progress.progress(1.0, text="Knowledge base ready.")
    return all_chunks, vector_store, video_titles


def render_sources(citations: list[str]) -> None:
    """Render source links underneath an answer."""
    if citations:
        st.markdown("**Sources**")
        st.markdown("\n".join(f"- {citation}" for citation in citations))


def render_messages() -> None:
    """Render prior conversation for the current browser session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("citations", []))


def ask_question(
    question: str, retrieval_mode: str, retrieval_k: int, fetch_k: int
) -> bool:
    """Rewrite a follow-up, retrieve relevant chunks, and store the answer."""
    try:
        standalone_question = rewrite_question(question, st.session_state.chat_history)
        retriever = create_retriever(
            st.session_state.vector_store,
            search_type=retrieval_mode,
            k=retrieval_k,
            mmr_fetch_k=fetch_k,
        )
        result = answer_question(retriever, standalone_question)
    except Exception as error:
        logger.exception("Question answering failed in Streamlit frontend")
        st.error("OpenAI could not answer this question. Check your API key and try again.")
        return False

    citations = format_source_citations(result.sources)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "citations": citations}
    )
    st.session_state.chat_history.append(
        ConversationTurn(question=question, answer=result.answer)
    )
    if standalone_question != question:
        st.session_state.messages[-1]["content"] = (
            f"_Searching for: {standalone_question}_\n\n{result.answer}"
        )
    return True


def summarize_processed_video() -> None:
    """Create a full summary from every processed chunk."""
    try:
        result = summarize_video(st.session_state.chunks)
    except Exception as error:
        logger.exception("Full-video summarization failed in Streamlit frontend")
        st.error("OpenAI could not summarize these videos. Check your API key and try again.")
        return

    st.session_state.messages.append({"role": "user", "content": "Summarize the video."})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": format_source_citations(result.sources),
        }
    )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="YouTube Knowledge Assistant", page_icon="Y", layout="wide")
    configure_logging()
    load_dotenv()
    initialize_session()

    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at top left, #f9edcf 0, #fbf8ef 44%, #dbece8 100%); }
        h1, h2, h3 { font-family: Georgia, 'Palatino Linotype', serif; color: #183a37; }
        [data-testid="stSidebar"] { background: #183a37; }
        [data-testid="stSidebar"] * { color: #f8f1df; }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: 0.5rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: #f8f1df !important;
            border: 2px solid #f8f1df !important;
            border-radius: 0.75rem !important;
            color: #183a37 !important;
            padding: 0.65rem 0.8rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label span { color: #183a37 !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #d65f3d !important;
            border-color: #d65f3d !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span { color: #ffffff !important; }
        .stButton > button { background: #d65f3d; color: white; border: 0; border-radius: 999px; font-weight: 700; }
        .stButton > button:hover { background: #a83e26; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("YouTube Knowledge Assistant")
    st.caption("Build a searchable knowledge base from public YouTube transcripts.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Add it to your local .env file, then restart Streamlit.")
        return

    with st.sidebar:
        st.header("Retrieval")
        retrieval_label = st.radio("Method", ["Similarity", "MMR"], horizontal=True)
        retrieval_mode = retrieval_label.lower()
        st.caption("Similarity picks the closest matching chunks. MMR mixes relevance with variety.")
        retrieval_k = st.slider("Chunks used for each answer", 1, 10, 4)
        st.caption("More chunks give the model more transcript context to work with.")
        fetch_k = st.slider("MMR candidate pool", retrieval_k, 30, max(12, retrieval_k))
        st.caption("MMR first looks at this many candidates, then keeps the most useful and diverse ones.")

    raw_urls = st.text_area(
        "Paste YouTube URL(s)",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID\nhttps://youtu.be/ANOTHER_VIDEO_ID",
        help="Use one public YouTube URL per line. Multiple videos become one knowledge base.",
        height=120,
    )
    if st.button("Process Videos", type="primary", use_container_width=True):
        try:
            video_ids = parse_urls(raw_urls)
            with st.spinner("Fetching transcripts and building the knowledge base..."):
                chunks, vector_store, titles = process_videos(video_ids)
        except ValueError as error:
            logger.warning("Invalid YouTube URL submitted: %s", error)
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))
        except Exception:
            logger.exception("Unexpected video-processing failure")
            st.error("The videos could not be processed. Please try again.")
        else:
            st.session_state.chunks = chunks
            st.session_state.vector_store = vector_store
            st.session_state.video_titles = titles
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.success(f"Knowledge base ready: {len(titles)} video(s), {len(chunks)} searchable chunks.")

    if st.session_state.vector_store is None:
        st.info("Process at least one video to begin asking questions.")
        return

    st.markdown("---")
    st.subheader("Ask about the videos")
    st.caption("Follow-up questions are rewritten for search using recent conversation context.")
    if st.button("Summarize All Processed Videos", use_container_width=True):
        with st.spinner("Summarizing every transcript chunk. This can take a while..."):
            summarize_processed_video()

    render_messages()
    question = st.chat_input("Ask a question about the processed videos")
    if question:
        with st.spinner("Searching the transcript knowledge base..."):
            answer_added = ask_question(question, retrieval_mode, retrieval_k, fetch_k)
        if answer_added:
            st.rerun()


if __name__ == "__main__":
    main()
