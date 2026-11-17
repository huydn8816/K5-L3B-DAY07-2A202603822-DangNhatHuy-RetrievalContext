import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_agent_benchmark import InitialSearchController
from src.ollama_controller import OllamaController


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class OllamaControllerTests(unittest.TestCase):
    def test_initial_search_uses_original_question_once(self):
        class Controller:
            def decide(self, question, history):
                return {"action": "answer", "answer": "answer", "citations": []}

        controller = InitialSearchController(Controller())
        self.assertEqual(controller.decide("original question", []), {
            "action": "search", "query": "original question",
        })
        self.assertEqual(controller.decide("original question", [{"context": "evidence"}]), {
            "action": "answer", "answer": "answer", "citations": [],
        })

    @patch("src.ollama_controller.urllib.request.urlopen")
    def test_maps_search_tool_call_and_sends_context(self, urlopen):
        urlopen.return_value = FakeResponse({
            "message": {"tool_calls": [{
                "function": {"name": "search", "arguments": {"query": "BM25"}},
            }]},
        })
        controller = OllamaController()

        result = controller.decide("How does BM25 rank?", [{
            "queries": ["rank function"],
            "context": "[c1]\nBM25 ranks documents.",
        }])

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(result, {"action": "search", "query": "BM25"})
        self.assertIn("BM25 ranks documents", body["messages"][1]["content"])
        self.assertIn("untrusted data", body["messages"][0]["content"])
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120)

    @patch("src.ollama_controller.urllib.request.urlopen")
    def test_rejects_non_tool_response(self, urlopen):
        urlopen.return_value = FakeResponse({"message": {"content": "unsupported"}})

        with self.assertRaisesRegex(ValueError, "đúng một tool call"):
            OllamaController().decide("question", [])


if __name__ == "__main__":
    unittest.main()
