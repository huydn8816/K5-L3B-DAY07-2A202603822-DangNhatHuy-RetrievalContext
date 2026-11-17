import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import DenseRetriever, HybridRetriever, LexicalRetriever


CHUNKS = [
    {"id": "c1", "text": "Dense retrieval uses embeddings.", "source_id": "doc1",
     "source": "doc1.md", "page": 1, "metadata": {"year": 2025}},
    {"id": "c2", "text": "BM25 ranks documents by term statistics.", "source_id": "doc2",
     "source": "doc2.md", "page": 2, "metadata": {"year": 2024}},
]


class FakeCollection:
    def __init__(self):
        self.call = None

    def query(self, **kwargs):
        self.call = kwargs
        return {
            "ids": [["c1"]],
            "documents": [[CHUNKS[0]["text"]]],
            "metadatas": [[{
                "source_id": "doc1", "source": "doc1.md", "page": 1,
                "metadata_json": '{"year": 2025}',
            }]],
            "distances": [[0.15]],
        }


class FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return self.chunks[:top_k]


class RetrievalTests(unittest.TestCase):
    def test_dense_retriever_encodes_and_preserves_metadata(self):
        collection = FakeCollection()
        retriever = DenseRetriever(collection, lambda texts: [[0.1, 0.2] for _ in texts])

        result = retriever.search("dense vectors", filters={"year": 2025}, top_k=2)

        self.assertEqual(collection.call["where"], {"meta_year": {"$eq": 2025}})
        self.assertEqual(collection.call["query_embeddings"], [[0.1, 0.2]])
        self.assertEqual(result[0]["metadata"], {"year": 2025})
        self.assertAlmostEqual(result[0]["score"], 0.85)
        self.assertEqual(result[0]["strategy"], "dense")

    def test_lexical_retriever_applies_filter_before_top_k(self):
        retriever = LexicalRetriever(CHUNKS)

        results = retriever.search("BM25 term statistics", filters={"year": 2024}, top_k=1)

        self.assertEqual([chunk["id"] for chunk in results], ["c2"])

    def test_hybrid_uses_rank_fusion_and_deduplicates(self):
        dense = FixedRetriever([CHUNKS[0], CHUNKS[1]])
        lexical = FixedRetriever([CHUNKS[1], CHUNKS[0]])
        retriever = HybridRetriever(dense, lexical)

        results = retriever.search("retrieval", top_k=2)

        self.assertEqual([chunk["id"] for chunk in results], ["c1", "c2"])
        self.assertEqual(results[0]["strategy"], "hybrid_rrf")
        self.assertAlmostEqual(results[0]["score"], results[1]["score"])
        self.assertEqual(dense.calls[0][2], 6)


if __name__ == "__main__":
    unittest.main()
