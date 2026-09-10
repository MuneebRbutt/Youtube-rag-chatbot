"""Regression coverage for the duration warning shown in the deployed UI."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.youtube import VideoMetadata


class DurationWarningUITests(unittest.TestCase):
    def test_long_video_shows_student_warning_without_fetching_captions(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-a-real-key"}),
            patch("src.youtube.fetch_video_metadata", return_value=VideoMetadata(
                "zjkBMFhNj_g", "[1hr Talk] Intro to Large Language Models", 3588,
            )),
            patch("src.youtube.fetch_english_transcript") as captions,
            patch("src.vectorstore.create_embeddings") as embeddings,
        ):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")).run()
            app.text_area[0].set_value("https://www.youtube.com/watch?v=zjkBMFhNj_g")
            app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("Spielberg", app.error[0].value)
        self.assertIn("student", app.error[0].value)
        self.assertIn("59:48", app.error[0].value)
        self.assertEqual(len(app.toast), 1)
        captions.assert_not_called()
        embeddings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
