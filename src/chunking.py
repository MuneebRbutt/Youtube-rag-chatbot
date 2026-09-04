"""Timestamp-aware transcript chunking."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_timestamp_aware_chunks(
    transcript_documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split a transcript while retaining the time range covered by each chunk."""
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
    video_title = transcript_documents[0].metadata["video_title"]
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
                        "video_title": video_title,
                        "start_time": overlapping_spans[0]["start_time"],
                        "end_time": overlapping_spans[-1]["end_time"],
                    },
                )
            )

    return timestamped_chunks
