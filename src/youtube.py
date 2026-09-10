"""YouTube URL parsing and transcript retrieval."""

import re
import json
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from youtube_transcript_api import YouTubeTranscriptApi

from src.limits import validate_video_duration


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    title: str
    duration_seconds: int


def fetch_video_metadata(video_id: str) -> VideoMetadata:
    """Read runtime from the watch page without requesting captions or video media."""
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID.")
    request = Request(
        f"https://www.youtube.com/watch?v={video_id}&hl=en",
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            page = response.read().decode("utf-8")
        # Decode a whole JSON object; a regex over nested braces can truncate it.
        marker = re.search(r'ytInitialPlayerResponse\s*=\s*', page)
        if marker is None:
            raise ValueError("Missing player metadata")
        player, _ = json.JSONDecoder().raw_decode(page[marker.end():])
        details = player.get("videoDetails", {})
        duration = int(details.get("lengthSeconds", 0))
        title = details.get("title", "").strip()
        if (
            details.get("videoId") != video_id
            or not title
            or duration <= 0
            or details.get("isLive")
            or details.get("isUpcoming")
        ):
            raise ValueError("Missing or unsupported video runtime")
        return VideoMetadata(video_id, title, duration)
    except Exception as error:
        raise RuntimeError(
            "YouTube did not provide a verifiable video duration. "
            "Processing stopped before fetching captions or using API credits. "
            "Please use a public, recorded video or try again later."
        ) from error


def preflight_videos(video_ids: list[str]) -> list[VideoMetadata]:
    """Validate every unique video's actual runtime before fetching any captions."""
    videos = []
    total_duration = 0
    for video_id in dict.fromkeys(video_ids):
        video = fetch_video_metadata(video_id)
        total_duration += video.duration_seconds
        validate_video_duration(video.title, video.duration_seconds, total_duration)
        videos.append(video)
    return videos


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


def fetch_video_title(video_id: str) -> str:
    """Fetch a public video title without requiring a YouTube Data API key."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    endpoint = f"https://www.youtube.com/oembed?{urlencode({'url': video_url, 'format': 'json'})}"
    request = Request(endpoint, headers={"User-Agent": "YouTube-RAG/1.0"})
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)

    title = str(payload.get("title", "")).strip()
    if not title:
        raise ValueError("YouTube did not return a video title.")
    return title


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
    if error_name in {"RequestBlocked", "IpBlocked"}:
        return (
            "YouTube is blocking caption requests from this app's server. "
            "This does not mean the video has no captions. "
            "Please try again later; no OpenAI credits were used for this submission."
        )
    if error_name == "PoTokenRequired":
        return "YouTube requires additional verification before this server can access captions."
    if error_name == "CouldNotRetrieveTranscript":
        return "YouTube could not provide the transcript right now. Please try again later."
    return "The video transcript could not be retrieved."
