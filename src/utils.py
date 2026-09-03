"""Shared application utilities."""

import logging


MAX_TRANSCRIPT_CHARACTERS = 1_000_000


def configure_logging() -> None:
    """Write diagnostic exceptions to a local log instead of the terminal."""
    logging.basicConfig(
        filename="rag_app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def show_error(message: str, next_step: str = "Please try again.") -> None:
    """Display a safe, actionable error without exposing internal details."""
    print(f"{message}\n{next_step}")
