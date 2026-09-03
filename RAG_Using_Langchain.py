import os
import re
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Extract and validate a video ID from a supported YouTube URL."""
    parsed_url = urlparse(url.strip())
    hostname = (parsed_url.hostname or "").lower()
    video_id: str | None = None

    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.path.startswith("/shorts/"):
            video_id = parsed_url.path.split("/", 3)[2]
    elif hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed_url.path.lstrip("/").split("/", 1)[0]

    if not video_id or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(
            "Enter a valid YouTube watch, short, or youtu.be URL with an 11-character video ID."
        )

    return video_id


def build_timestamp_aware_chunks(
    transcript_documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk a transcript while retaining the time range each chunk covers."""
    if not transcript_documents:
        return []

    merged_parts: list[str] = []
    segment_spans: list[dict[str, float | int]] = []
    cursor = 0

    for document in transcript_documents:
        text = document.page_content.strip()
        if not text:
            continue

        if merged_parts:
            merged_parts.append("\n")
            cursor += 1

        start_char = cursor
        merged_parts.append(text)
        cursor += len(text)
        segment_spans.append(
            {
                "start_char": start_char,
                "end_char": cursor,
                "start_time": float(document.metadata["start_time"]),
                "end_time": float(document.metadata["end_time"]),
            }
        )

    if not segment_spans:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    video_id = transcript_documents[0].metadata["video_id"]
    raw_chunks = splitter.create_documents(
        ["".join(merged_parts)],
        metadatas=[{"video_id": video_id}],
    )

    timestamped_chunks: list[Document] = []
    for chunk in raw_chunks:
        start_char = chunk.metadata["start_index"]
        end_char = start_char + len(chunk.page_content)
        overlapping_spans = [
            span
            for span in segment_spans
            if span["start_char"] < end_char and span["end_char"] > start_char
        ]

        if overlapping_spans:
            timestamped_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "video_id": video_id,
                        "start_time": overlapping_spans[0]["start_time"],
                        "end_time": overlapping_spans[-1]["end_time"],
                    },
                )
            )

    return timestamped_chunks


def load_transcript(video_id: str) -> list[Document]:
    """Fetch English captions and convert each caption segment into a document."""
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id, languages=["en"])

    documents: list[Document] = []
    for segment in transcript:
        text = segment.text.strip()
        if text:
            start_time = float(segment.start)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": start_time + float(segment.duration),
                    },
                )
            )

    return documents


def format_docs(retrieved_docs: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in retrieved_docs)


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing. Add it to .env and run the script again.")
        return

    video_url = "https://www.youtube.com/watch?v=7xTGNNLPyMI"  # Paste a YouTube URL here.
    try:
        video_id = extract_video_id(video_url)
    except ValueError as error:
        print(error)
        return

    try:
        transcript_documents = load_transcript(video_id)
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return

    if not transcript_documents:
        print("No transcript segments were returned for this video.")
        return

    print(f"Loaded {len(transcript_documents)} transcript segments.")
    chunks = build_timestamp_aware_chunks(transcript_documents)
    if not chunks:
        print("Could not create chunks from the transcript.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"],
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "Can you summarize the video?"
    print(rag_chain.invoke(question))


if __name__ == "__main__":
    main()
