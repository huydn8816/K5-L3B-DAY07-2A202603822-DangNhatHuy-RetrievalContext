import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fetch_wikimedia import ARTICLES, make_chunks


class WikimediaCorpusTests(unittest.TestCase):
    def test_curated_source_list_has_thirty_unique_articles(self):
        self.assertEqual(len(ARTICLES), 30)
        self.assertEqual(len(set(ARTICLES)), 30)

    def test_sections_and_attribution_survive_chunking(self):
        article = {
            "title": "Example article",
            "source_id": "wikipedia:123",
            "source": "https://en.wikipedia.org/wiki/Example",
            "revision_id": 456,
            "revision_url": "https://en.wikipedia.org/w/index.php?oldid=456",
            "text": "Lead text.\n\n== Ranking ==\nRanks relevant documents.",
        }

        chunks = make_chunks(article, max_chars=50, overlap=5)

        self.assertEqual([chunk["page"] for chunk in chunks], [1, 2])
        self.assertEqual([chunk["metadata"]["section"] for chunk in chunks], ["Lead", "Ranking"])
        self.assertTrue(all(chunk["metadata"]["license"] == "CC BY-SA 4.0" for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["revision_url"] == article["revision_url"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
