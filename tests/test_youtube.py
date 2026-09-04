import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.chunking import build_timestamp_aware_chunks
from src.ingestion import transcript_to_documents
from src.rag import (
    ConversationTurn,
    batch_summaries,
    format_source_citations,
    format_timestamp,
    rewrite_question,
    summarize_video,
)
from src.retrieval import create_retriever
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


class CitationTests(unittest.TestCase):
    def test_timestamp_formatting(self):
        self.assertEqual(format_timestamp(1122), "18:42")
        self.assertEqual(format_timestamp(3723), "1:02:03")

    def test_citations_link_to_video_timestamp(self):
        source = Document(
            page_content="Transcript text",
            metadata={
                "video_id": "VMj-3S1tku0",
                "video_title": "Backpropagation Explained",
                "start_time": 1122.7,
            },
        )
        self.assertEqual(
            format_source_citations([source, source]),
            [
                "[Backpropagation Explained at 18:42](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1122s)"
            ],
        )

    def test_citations_identify_multiple_videos(self):
        sources = [
            Document(
                page_content="First video",
                metadata={
                    "video_id": "VMj-3S1tku0",
                    "video_title": "Video One",
                    "start_time": 60,
                },
            ),
            Document(
                page_content="Second video",
                metadata={
                    "video_id": "7xTGNNLPyMI",
                    "video_title": "Video Two",
                    "start_time": 120,
                },
            ),
        ]
        self.assertEqual(
            format_source_citations(sources),
            [
                "[Video One at 1:00](https://www.youtube.com/watch?v=VMj-3S1tku0&t=60s)",
                "[Video Two at 2:00](https://www.youtube.com/watch?v=7xTGNNLPyMI&t=120s)",
            ],
        )


class IngestionTests(unittest.TestCase):
    def test_video_title_is_preserved_in_timestamped_chunks(self):
        segment = type(
            "Segment",
            (),
            {"text": "A transcript segment.", "start": 42, "duration": 3},
        )()
        documents = transcript_to_documents(
            [segment],
            video_id="VMj-3S1tku0",
            video_title="Backpropagation Explained",
        )
        chunks = build_timestamp_aware_chunks(documents)

        self.assertEqual(chunks[0].metadata["video_id"], "VMj-3S1tku0")
        self.assertEqual(
            chunks[0].metadata["video_title"], "Backpropagation Explained"
        )


class SummaryBatchTests(unittest.TestCase):
    def test_partial_summaries_are_batched_without_dropping_text(self):
        batches = batch_summaries(["first", "second", "third"], max_characters=11)
        self.assertEqual(batches, ["first\n\nsecond", "third"])

    def test_full_summary_maps_every_transcript_chunk(self):
        mapped_chunks = []

        class MapChain:
            def invoke(self, values):
                mapped_chunks.append(values["chunk"])
                return f"Summary: {values['chunk']}"

        class ReduceChain:
            def invoke(self, values):
                return values["summaries"]

        chunks = [
            Document(
                page_content=text,
                metadata={"video_id": "VMj-3S1tku0", "start_time": index},
            )
            for index, text in enumerate(["first", "second", "third"])
        ]

        with patch("src.rag.create_summary_chains", return_value=(MapChain(), ReduceChain())):
            summarize_video(chunks)

        self.assertEqual(mapped_chunks, ["first", "second", "third"])


class RetrievalTests(unittest.TestCase):
    class FakeVectorStore:
        def __init__(self):
            self.arguments = None

        def as_retriever(self, **kwargs):
            self.arguments = kwargs
            return "retriever"

    def test_similarity_is_the_default(self):
        vector_store = self.FakeVectorStore()
        self.assertEqual(create_retriever(vector_store), "retriever")
        self.assertEqual(
            vector_store.arguments,
            {"search_type": "similarity", "search_kwargs": {"k": 4}},
        )


class ConversationTests(unittest.TestCase):
    def test_follow_up_uses_recent_history_for_rewriting(self):
        captured_values = {}

        class RewriteChain:
            def invoke(self, values):
                captured_values.update(values)
                return "Why is backpropagation important?"

        history = [
            ConversationTurn(question="What is backpropagation?", answer="It trains a model."),
            ConversationTurn(question="What does it calculate?", answer="Gradients."),
        ]
        with patch("src.rag.create_question_rewriter", return_value=RewriteChain()):
            rewritten = rewrite_question(
                "Why is it important?",
                history,
                max_history_turns=1,
                max_history_characters=100,
            )

        self.assertEqual(rewritten, "Why is backpropagation important?")
        self.assertNotIn("What is backpropagation?", captured_values["history"])
        self.assertIn("What does it calculate?", captured_values["history"])

    def test_first_question_is_not_rewritten(self):
        self.assertEqual(rewrite_question("What is backpropagation?", []), "What is backpropagation?")

    def test_mmr_uses_a_larger_candidate_pool(self):
        vector_store = self.FakeVectorStore()
        create_retriever(vector_store, search_type="mmr", k=4, mmr_fetch_k=12)
        self.assertEqual(
            vector_store.arguments,
            {"search_type": "mmr", "search_kwargs": {"k": 4, "fetch_k": 12}},
        )


if __name__ == "__main__":
    unittest.main()
