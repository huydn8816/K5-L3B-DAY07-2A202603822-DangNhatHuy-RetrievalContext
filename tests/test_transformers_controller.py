import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transformers_controller import TransformersController


class TransformersControllerTests(unittest.TestCase):
    def test_parses_search_and_final_answer_tool_calls(self):
        search = TransformersController._parse_tool_call(
            '<tool_call>\n{"name":"search","arguments":{"query":"RAG retrieval"}}\n</tool_call>'
        )
        answer = TransformersController._parse_tool_call(
            '<tool_call>\n{"name":"final_answer","arguments":{"answer":"RAG uses [c1].","citations":["c1"]}}\n</tool_call>'
        )

        self.assertEqual(search, {"action": "search", "query": "RAG retrieval"})
        self.assertEqual(answer, {
            "action": "answer", "answer": "RAG uses [c1].", "citations": ["c1"],
        })

    def test_extracts_citations_explicitly_present_in_answer(self):
        answer = TransformersController._parse_tool_call(
            '<tool_call>{"name":"final_answer","arguments":{"answer":"Evidence [c1].","citations":[]}}</tool_call>'
        )
        self.assertEqual(answer["citations"], ["c1"])

    def test_does_not_turn_explicit_abstention_into_a_cited_answer(self):
        answer = TransformersController._normalize_answer(
            "INSUFFICIENT EVIDENCE [c1]", ["c1"], ["c1"], {"c1": "some evidence"},
        )
        self.assertEqual(answer["citations"], [])

    def test_normalizes_model_citations_only_when_the_id_is_in_context(self):
        answer = TransformersController._normalize_answer(
            "The result is relevant.",
            [["wiki-15271-p1-c1", "Information retrieval | page 1"], "unretrieved-id"],
            ["wiki-15271-p1-c1"],
        )
        self.assertEqual(answer, {
            "action": "answer",
            "answer": "The result is relevant. [wiki-15271-p1-c1]",
            "citations": ["wiki-15271-p1-c1"],
            "citation_method": "model",
        })

    def test_links_uncited_answer_only_to_high_overlap_context(self):
        context = {
            "c1": "Information retrieval is the task of identifying and retrieving information system resources relevant to an information need.",
            "c2": "A vector database stores embeddings.",
        }
        linked = TransformersController._normalize_answer(
            "Information retrieval identifies and retrieves resources relevant to an information need.",
            [], ["c1", "c2"], context,
        )
        unsupported = TransformersController._normalize_answer(
            "It was invented in 1950.", [], ["c1", "c2"], context,
        )

        self.assertEqual(linked["citations"], ["c1"])
        self.assertEqual(linked["citation_method"], "lexical_context_overlap")
        self.assertIn("[c1]", linked["answer"])
        self.assertEqual(unsupported["citations"], [])

    def test_extracts_answer_from_truncated_structured_response(self):
        response = (
            '{\n  "answer": "Relevant resources for an information need.",\n'
            '  "citations": [{"chunk-id": "wiki-15271-p1-c1", "citation": "[wiki-15271-p1-c1]"}'
        )
        answer = TransformersController._parse_answer_response(
            response, ["wiki-15271-p1-c1"],
        )
        self.assertEqual(answer, {
            "action": "answer",
            "answer": "Relevant resources for an information need. [wiki-15271-p1-c1]",
            "citations": ["wiki-15271-p1-c1"],
            "citation_method": "model",
        })

    def test_rejects_plain_text_without_tool_call(self):
        with self.assertRaisesRegex(ValueError, "did not return a tool call"):
            TransformersController._parse_tool_call("I think the answer is RAG.")

    def test_rejects_malformed_tool_arguments(self):
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            TransformersController._parse_tool_call(
                '<tool_call>{"name":"search","arguments":{"query":" "}}</tool_call>'
            )


if __name__ == "__main__":
    unittest.main()
