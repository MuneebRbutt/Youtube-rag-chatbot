"""YouTube URL parsing and transcript retrieval."""

import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Extract and validate an ID from a supported YouTube URL."""
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


def fetch_english_transcript(video_id: str):
    """Fetch raw English transcript segments for a validated video ID."""
    return YouTubeTranscriptApi().fetch(video_id, languages=["en"])


def transcript_error_message(error: Exception) -> str:
    """Map provider exceptions to safe messages for normal users."""
    error_name = type(error).__name__

    if error_name == "InvalidVideoId":
        return "The YouTube URL contains an invalid video ID."
    if error_name in {"VideoUnavailable", "VideoUnplayable"}:
        return "This video is unavailable, private, deleted, or cannot be played."
    if error_name == "TranscriptsDisabled":
        return "Captions are disabled for this video."
    if error_name in {"NoTranscriptFound", "NotTranslatable", "TranslationLanguageNotAvailable"}:
        return "No English transcript was found for this video."
    if error_name in {"CouldNotRetrieveTranscript", "RequestBlocked", "IpBlocked", "PoTokenRequired"}:
        return "The transcript is currently unavailable."
    return "The video transcript could not be retrieved."
