"""Unit tests for the advanced PDF pipeline and the Claude provider (no network calls)."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from src.anthropic_controller import AnthropicController, load_api_key
from src.build_pdf_corpus import CAPTION, split_text
from src.evaluate_advanced_retrieval import requirements, satisfied


def gold(page, evidence, source_id="thesis"):
    return {"source_id": source_id, "page": page, "evidence": evidence}


class SplitTextTests(unittest.TestCase):
    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(split_text("   \n  "), [])

    def test_long_text_is_split_with_overlap_and_covers_everything(self):
        text = "a" * 1000 + " " + "b" * 300
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][-100:], chunks[1][:100])  # overlap of 100 characters
        self.assertTrue(chunks[-1].endswith(text[-50:]))
        self.assertEqual(chunks[0] + "".join(c[100:] for c in chunks[1:]), text)  # no gaps, overlap = 100

    def test_caption_pattern_matches_tables_and_figures_only(self):
        self.assertTrue(CAPTION.match("Bảng 3.5: Mô tả tổng quan"))
        self.assertTrue(CAPTION.match("Hình 2.1: Ví dụ"))
        self.assertFalse(CAPTION.match("Theo bảng 3.5 thì"))


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"id": "c1", "page": 65, "source_id": "thesis", "text": "LED2 Bộ đèn LED M36 1200/40W 6500K Bộ đèn M16, M26, M36"},
            {"id": "c2", "page": 67, "source_id": "thesis", "text": "Xác định 72 điểm bất thường của bộ dữ liệu trong tổng số 1078 điểm"},
            {"id": "c3", "page": 68, "source_id": "thesis", "text": "Biểu đồ (d) LED4 gần như toàn bộ giá trị tập trung gần 0"},
        ]

    def test_single_evidence_is_satisfied_by_any_alternative(self):
        q = {"gold": [gold(65, "LED2 Bộ đèn LED M36")]}
        self.assertTrue(satisfied(self.chunks[:1], requirements(q)))
        self.assertFalse(satisfied(self.chunks[2:], requirements(q)))

    def test_multihop_needs_every_required_evidence(self):
        q = {"gold_all": [gold(65, "LED2 Bộ đèn LED M36"), gold(67, "72 điểm bất thường")]}
        self.assertFalse(satisfied(self.chunks[:1], requirements(q)))
        self.assertTrue(satisfied(self.chunks[:2], requirements(q)))

    def test_page_must_match_not_only_text(self):
        q = {"gold": [gold(99, "LED2 Bộ đèn LED M36")]}
        self.assertFalse(satisfied(self.chunks[:1], requirements(q)))


class AnthropicControllerTests(unittest.TestCase):
    def _response(self, blocks):
        return {"content": blocks}

    def test_missing_key_is_reported_without_revealing_value(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Thiếu ANTHROPIC_API_KEY"):
                load_api_key(Path(tmp) / ".env")

    def test_key_is_read_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=True):
            env = Path(tmp) / ".env"
            env.write_text("ANTHROPIC_API_KEY=sk-test-123\n", encoding="utf-8")
            self.assertEqual(load_api_key(env), "sk-test-123")

    def test_search_tool_use_is_normalized(self):
        controller = AnthropicController(api_key="sk-test")
        block = {"type": "tool_use", "name": "search", "input": {"query": "LED2"}}
        with mock.patch.object(controller, "_post", return_value=self._response([block])):
            self.assertEqual(controller.decide("q", []), {"action": "search", "query": "LED2"})

    def test_final_answer_tool_use_is_normalized(self):
        controller = AnthropicController(api_key="sk-test")
        block = {"type": "tool_use", "name": "final_answer",
                 "input": {"answer": "Có [c1]", "citations": ["c1"]}}
        with mock.patch.object(controller, "_post", return_value=self._response([block])):
            self.assertEqual(controller.decide("q", []),
                             {"action": "answer", "answer": "Có [c1]", "citations": ["c1"]})

    def test_zero_or_two_tool_uses_are_rejected(self):
        controller = AnthropicController(api_key="sk-test")
        with mock.patch.object(controller, "_post", return_value=self._response([{"type": "text", "text": "x"}])):
            with self.assertRaises(ValueError):
                controller.decide("q", [])

    def test_http_error_reports_status_only(self):
        controller = AnthropicController(api_key="sk-secret-value")
        error = urllib.error.HTTPError("url", 401, "unauthorized", {}, io.BytesIO(b""))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as ctx:
                controller._post(mock.Mock())
            self.assertIn("401", str(ctx.exception))
            self.assertNotIn("sk-secret-value", str(ctx.exception))

    def test_request_disables_parallel_tool_calls(self):
        controller = AnthropicController(api_key="sk-test")
        captured = {}

        def fake_post(request):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return self._response([{"type": "tool_use", "name": "search", "input": {"query": "x"}}])

        with mock.patch.object(controller, "_post", side_effect=fake_post):
            controller.decide("q", [])
        self.assertTrue(captured["payload"]["tool_choice"]["disable_parallel_tool_use"])


if __name__ == "__main__":
    unittest.main()
