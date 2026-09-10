"""Convert transcript segments into LangChain documents."""

from collections.abc import Iterable

from langchain_core.documents import Document


def transcript_to_documents(
    transcript: Iterable, video_id: str, video_title: str
) -> list[Document]:
    """Attach video and timestamp metadata to non-empty transcript segments."""
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
                        "video_title": video_title,
                        "start_time": start_time,
                        "end_time": start_time + float(segment.duration),
                    },
                )
            )
    return documents


def transcript_character_count(documents: Iterable[Document]) -> int:
    """Return the total text size used to guard costly embedding operations."""
    return sum(len(document.page_content) for document in documents)


def transcript_duration_seconds(documents: Iterable[Document]) -> float:
    """Estimate processed video duration from the final caption end time."""
    return max(
        (float(document.metadata.get("end_time", 0)) for document in documents),
        default=0.0,
    )
