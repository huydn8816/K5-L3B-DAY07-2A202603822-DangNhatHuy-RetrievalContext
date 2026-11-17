import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate_retrieval import evaluate_method


CHUNK = {
    "id": "c1", "text": "Relevant evidence.", "source_id": "doc",
    "source": "doc.pdf", "page": 1, "metadata": {}, "score": 0.9,
}


class Retriever:
    def search(self, query, *, filters=None, top_k=3):
        return [CHUNK]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_reports_common_topk_and_context_evidence_metrics(self):
        question = {
            "id": "q1", "split": "dev", "question": "What?",
            "gold": [{"source_id": "doc", "page": 1, "evidence": "Relevant evidence"}],
        }

        summary, traces = evaluate_method("fake", Retriever(), [question], index_seconds=0)

        self.assertEqual(summary["hit1"], {"n": 1, "N": 1, "rate": 1.0})
        self.assertEqual(summary["evidence_hit3"], {"n": 1, "N": 1, "rate": 1.0})
        self.assertEqual(summary["context_evidence_rate"], {"n": 1, "N": 1, "rate": 1.0})
        self.assertTrue(traces[0]["results"][0]["hit"])

    def test_rejects_duplicate_chunk_ids(self):
        class DuplicateRetriever:
            def search(self, query, *, filters=None, top_k=3):
                return [CHUNK, CHUNK]

        with self.assertRaisesRegex(ValueError, "duplicate chunk IDs"):
            evaluate_method("bad", DuplicateRetriever(), [{
                "id": "q1", "split": "dev", "question": "What?",
                "gold": [{"source_id": "doc", "page": 1, "evidence": "Relevant evidence"}],
            }], index_seconds=0)


if __name__ == "__main__":
    unittest.main()
