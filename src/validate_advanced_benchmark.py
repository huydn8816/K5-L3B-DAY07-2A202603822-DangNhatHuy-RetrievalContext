"""Check the advanced benchmark against the built corpus before any retrieval run.

- every gold evidence must be an exact substring of a chunk on the stated page;
- unanswerable questions must have absence terms that do not occur in the PDF text;
- split and group counts must match ADVANCED.md (dev: 15+3, test: 24 answerable by group + 6 unanswerable).
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/advanced"))
    args = parser.parse_args()

    questions = json.loads((args.data / "benchmark_questions.json").read_text(encoding="utf-8"))
    chunks = json.loads((args.data / "chunks.json").read_text(encoding="utf-8"))
    pages = json.loads((args.data / "pages.json").read_text(encoding="utf-8"))
    full_text = norm(" ".join(p["text"] for p in pages)).lower()
    errors = []

    for q in questions:
        evidence_sets = []
        if q["answerable"]:
            if q.get("gold"):
                evidence_sets.append(("gold(any)", q["gold"]))
            for g in q.get("gold_all", []):
                evidence_sets.append(("gold_all", [g]))
            if not evidence_sets:
                errors.append(f"{q['id']}: answerable without gold")
        else:
            for term in q.get("absence_terms", []):
                if term.lower() in full_text:
                    errors.append(f"{q['id']}: absence term {term!r} appears in the PDF text")
        for label, golds in evidence_sets:
            for g in golds:
                ev = norm(g["evidence"])
                ok = any(c["page"] == g["page"] and ev in norm(c["text"]) for c in chunks)
                if not ok:
                    errors.append(f"{q['id']}: {label} evidence not found on page {g['page']}: {ev[:60]!r}")

    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    summary = {
        "dev_answerable": sum(q["answerable"] for q in dev),
        "dev_unanswerable": sum(not q["answerable"] for q in dev),
        "test_answerable": sum(q["answerable"] for q in test),
        "test_unanswerable": sum(not q["answerable"] for q in test),
        "test_groups": dict(Counter(q["group"] for q in test if q["answerable"])),
        "dev_groups": dict(Counter(q["group"] for q in dev if q["answerable"])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    required = {"test_answerable": 24, "test_unanswerable": 6, "dev_answerable": 15, "dev_unanswerable": 3}
    for key, value in required.items():
        if summary[key] != value:
            errors.append(f"count {key}={summary[key]} expected {value}")
    required_groups = {"table": 6, "image": 5, "chart": 5, "multihop": 4, "text": 4}
    for group, value in required_groups.items():
        if summary["test_groups"].get(group, 0) != value:
            errors.append(f"test group {group}={summary['test_groups'].get(group, 0)} expected {value}")

    if errors:
        print("\n".join(["ERRORS:"] + errors))
        return 1
    print("OK: all evidence matches chunks and counts match ADVANCED.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
