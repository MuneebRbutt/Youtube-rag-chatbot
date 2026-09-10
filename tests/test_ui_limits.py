"""Regression coverage for the duration warning shown in the deployed UI."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

class DurationWarningUITests(unittest.TestCase):
    def test_long_transcript_shows_student_warning_before_embedding(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-a-real-key"}),
            patch("src.youtube.fetch_video_title", return_value="[1hr Talk] Intro to Large Language Models"),
            patch("src.youtube.fetch_english_transcript", return_value=[
                SimpleNamespace(text="Final caption", start=3580, duration=8),
            ]) as captions,
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
        captions.assert_called_once_with("zjkBMFhNj_g")
        embeddings.assert_not_called()

    def test_short_video_processes_without_watch_page_and_survives_rejected_replacement(self):
        store = object()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-a-real-key"}),
            patch("src.youtube.urlopen", side_effect=AssertionError("Unexpected watch-page lookup")),
            patch("src.youtube.fetch_video_title", return_value="Demo video"),
            patch("src.youtube.fetch_english_transcript", side_effect=[
                [SimpleNamespace(text="A valid transcript", start=1199, duration=1)],
                [SimpleNamespace(text="An over-limit transcript", start=1200, duration=1)],
            ]),
            patch("src.vectorstore.create_embeddings", return_value=(object(), [[0.1]])) as embeddings,
            patch("src.vectorstore.create_vector_store", return_value=store),
        ):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")).run()
            app.text_area[0].set_value("https://youtu.be/VMj-3S1tku0")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual(len(app.success), 1)
            self.assertIs(app.session_state["vector_store"], store)

            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Spielberg", app.error[0].value)
            self.assertIs(app.session_state["vector_store"], store)

        embeddings.assert_called_once()


if __name__ == "__main__":
    unittest.main()
