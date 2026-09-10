"""Cost-control limits for video ingestion."""


MAX_VIDEO_DURATION_SECONDS = 20 * 60
MAX_TOTAL_DURATION_SECONDS = 30 * 60


def format_duration(seconds: float) -> str:
    """Format a duration, rounding up so an overage is never understated."""
    total_seconds = max(0, int(seconds))
    if seconds > total_seconds:
        total_seconds += 1
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


class VideoDurationLimitError(ValueError):
    """Raised when one video or a submitted collection exceeds the demo quota."""

    def __init__(
        self, message: str, *, duration_seconds: float, limit_seconds: float
    ) -> None:
        super().__init__(message)
        self.duration_seconds = duration_seconds
        self.limit_seconds = limit_seconds


def validate_video_duration(
    video_title: str,
    video_duration_seconds: float,
    combined_duration_seconds: float,
) -> None:
    """Enforce the per-video and combined duration limits for one submission."""
    if video_duration_seconds > MAX_VIDEO_DURATION_SECONDS:
        raise VideoDurationLimitError(
            (
                f'"{video_title}" is about {format_duration(video_duration_seconds)} long. '
                f"Each video must be {format_duration(MAX_VIDEO_DURATION_SECONDS)} or shorter."
            ),
            duration_seconds=video_duration_seconds,
            limit_seconds=MAX_VIDEO_DURATION_SECONDS,
        )

    if combined_duration_seconds > MAX_TOTAL_DURATION_SECONDS:
        raise VideoDurationLimitError(
            (
                f"These videos add up to about {format_duration(combined_duration_seconds)}. "
                f"The combined limit is {format_duration(MAX_TOTAL_DURATION_SECONDS)}."
            ),
            duration_seconds=combined_duration_seconds,
            limit_seconds=MAX_TOTAL_DURATION_SECONDS,
        )
