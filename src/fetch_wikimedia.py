"""Build a cited, reproducible corpus from selected English Wikipedia articles."""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ARTICLES = [
    "Information retrieval",
    "Search engine",
    "Vector space model",
    "Tf–idf",
    "Okapi BM25",
    "PageRank",
    "Learning to rank",
    "Inverted index",
    "Precision and recall",
    "Mean reciprocal rank",
    "Discounted cumulative gain",
    "Relevance feedback",
    "Query expansion",
    "Natural language processing",
    "Word embedding",
    "Sentence embedding",
    "Transformer (deep learning)",
    "BERT (language model)",
    "Web crawler",
    "Semantic search",
    "Vector database",
    "Retrieval-augmented generation",
    "Prompt engineering",
    "Information extraction",
    "Knowledge graph",
    "Cosine similarity",
    "Clustering",
    "Document classification",
    "Question answering",
    "Text mining",
]

API_URL = "https://en.wikipedia.org/w/api.php"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
USER_AGENT = "Day07-RAG-Lab/1.0 (educational corpus builder)"


def fetch_article(title: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "exlimit": "1",
        "titles": title,
    })
    request = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                wait_seconds = max(float(retry_after), 5) if retry_after else max(min(2 ** attempt, 8), 5)
            except ValueError:
                wait_seconds = max(min(2 ** attempt, 8), 5)
            time.sleep(wait_seconds)
    pages = data.get("query", {}).get("pages", {})
    article = next(iter(pages.values()), {})
    extract = article.get("extract", "").strip()
    if "missing" in article or not extract:
        raise ValueError(f"Wikipedia article is missing or has no extract: {title}")
    return {
        "title": article["title"],
        "source_id": f"wikipedia:{article['pageid']}",
        "source": article["fullurl"],
        "revision_id": article["lastrevid"],
        "revision_url": f"{article['fullurl']}?oldid={article['lastrevid']}",
        "text": extract,
    }


def _split_section(text: str, *, max_chars: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def make_chunks(article: dict[str, Any], *, max_chars: int = 1200, overlap: int = 150) -> list[dict[str, Any]]:
    sections: list[tuple[str, list[str]]] = [("Lead", [])]
    for line in article["text"].splitlines():
        heading = re.fullmatch(r"={2,}\s*(.*?)\s*={2,}", line.strip())
        if heading:
            sections.append((heading.group(1).strip(), []))
        elif line.strip():
            sections[-1][1].append(line.strip())

    chunks = []
    for page, (section, paragraphs) in enumerate(sections, start=1):
        section_text = "\n".join(paragraphs)
        for part, text in enumerate(_split_section(section_text, max_chars=max_chars, overlap=overlap), start=1):
            chunks.append({
                "id": f"wiki-{article['source_id'].split(':')[1]}-p{page}-c{part}",
                "text": text,
                "source_id": article["source_id"],
                "source": article["source"],
                "page": page,
                "metadata": {
                    "title": article["title"],
                    "section": section,
                    "part": part,
                    "language": "en",
                    "revision_id": article["revision_id"],
                    "revision_url": article["revision_url"],
                    "license": "CC BY-SA 4.0",
                    "license_url": LICENSE_URL,
                },
            })
    return chunks


def build_corpus(cache_dir: Path, *, delay: float = 5.0) -> list[dict[str, Any]]:
    chunks = []
    failures = []
    for title in ARTICLES:
        cache_path = cache_dir / (re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-") + ".json")
        cached = cache_path.exists()
        try:
            if cached:
                article = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                article = fetch_article(title)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(article, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            chunks.extend(make_chunks(article))
            print(f"Fetched {article['title']}: {sum(c['source_id'] == article['source_id'] for c in chunks)} chunks")
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{title}: {error}")
        if not cached:
            time.sleep(delay)
    if failures:
        raise RuntimeError("Corpus incomplete:\n" + "\n".join(failures))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the attributed Wikimedia retrieval corpus")
    parser.add_argument("--output", type=Path, default=Path("data/wikimedia_chunks.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.wikimedia_cache"))
    parser.add_argument("--delay", type=float, default=5.0, help="seconds between API requests")
    args = parser.parse_args()
    chunks = build_corpus(args.cache_dir, delay=args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    articles = {
        (chunk["source_id"], chunk["metadata"]["revision_id"]): (
            chunk["metadata"]["title"], chunk["metadata"]["revision_url"]
        )
        for chunk in chunks
    }
    attribution = [
        "Wikipedia text is used under CC BY-SA 4.0.",
        f"License: {LICENSE_URL}",
        "Each article link identifies the source and exact revision; article histories identify contributors.",
        "",
    ]
    attribution.extend(f"- {title}: {url}" for title, url in articles.values())
    args.output.with_name("WIKIMEDIA_ATTRIBUTION.txt").write_text(
        "\n".join(attribution) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(chunks)} chunks from {len(ARTICLES)} articles to {args.output}")


if __name__ == "__main__":
    main()
