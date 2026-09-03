import unittest

from src.youtube import extract_video_id, transcript_error_message


class YouTubeUrlTests(unittest.TestCase):
    def test_supported_url_formats(self):
        video_id = "VMj-3S1tku0"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://youtu.be/{video_id}",
            f"https://www.youtube.com/shorts/{video_id}",
        ]

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), video_id)

    def test_invalid_url_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/watch?v=VMj-3S1tku0")

    def test_transcript_errors_have_safe_messages(self):
        error = type("NoTranscriptFound", (Exception,), {})()
        self.assertEqual(
            transcript_error_message(error),
            "No English transcript was found for this video.",
        )


if __name__ == "__main__":
    unittest.main()
