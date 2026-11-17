"""Retrieval evaluation for the advanced PDF benchmark.

Single-evidence questions: a hit means any gold evidence appears in the top-k chunks.
Multi-hop questions (gold_all): a hit means every required evidence appears in the top-k chunks.
Context Evidence Rate uses the same rule on the chunks actually packed into the context budget.
Unanswerable questions are excluded from all retrieval metrics.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from starter.baseline import TfidfIndex, hit, pack_context
from src.retrieval import HybridRetriever, LexicalRetriever, create_dense_retriever

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def requirements(question: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Each inner list is a set of alternatives; every inner list must be satisfied."""
    if question.get("gold_all"):
        return [[g] for g in question["gold_all"]]
    return [question["gold"]]


def satisfied(chunks: list[dict[str, Any]], reqs: list[list[dict[str, Any]]]) -> bool:
    return all(any(hit(c, alternatives) for c in chunks) for alternatives in reqs)


def rate(n: int, total: int) -> dict[str, Any]:
    return {"n": n, "N": total, "rate": n / total if total else None}


def evaluate(name: str, retriever: Any, questions: list[dict[str, Any]], *, top_k: int, budget: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces, times = [], []
    for question in questions:
        started = time.perf_counter()
        results = retriever.search(question["question"], filters=question.get("filter"), top_k=top_k)
        times.append((time.perf_counter() - started) * 1000)
        context, context_ids = pack_context(results, budget=budget)
        reqs = requirements(question)
        traces.append({
            "query_id": question["id"],
            "group": question["group"],
            "multi_hop": bool(question.get("gold_all")),
            "question": question["question"],
            "results": [{"id": c["id"], "page": c["page"], "rank": i + 1,
                         "score": c.get("score"), "hit": any(hit(c, a) for a in reqs)}
                        for i, c in enumerate(results)],
            "context_ids": context_ids,
            "hit1": satisfied(results[:1], reqs),
            "hit3": satisfied(results[:3], reqs),
            "hitk": satisfied(results, reqs),
            "context_hit": satisfied([c for c in results if c["id"] in context_ids], reqs),
            "gold_pages": sorted({g["page"] for r in reqs for g in r}),
            "elapsed_ms": times[-1],
        })
    single = [t for t in traces if not t["multi_hop"]]
    summary = {
        "strategy": name,
        "n": len(traces),
        "hit1_single_only": rate(sum(t["hit1"] for t in single), len(single)),
        "hit3_all_questions": rate(sum(t["hit3"] for t in traces), len(traces)),
        "hitk_all_questions": rate(sum(t["hitk"] for t in traces), len(traces)),
        "context_evidence_rate": rate(sum(t["context_hit"] for t in traces), len(traces)),
        "by_group": {},
        "median_query_ms": statistics.median(times) if times else None,
    }
    for group in sorted({t["group"] for t in traces}):
        subset = [t for t in traces if t["group"] == group]
        summary["by_group"][group] = {
            "n": len(subset),
            "hit3": rate(sum(t["hit3"] for t in subset), len(subset)),
            "context_evidence_rate": rate(sum(t["context_hit"] for t in subset), len(subset)),
        }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced PDF retrieval benchmark (TF-IDF, dense, hybrid RRF)")
    parser.add_argument("--chunks", type=Path, default=Path("data/advanced/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/advanced/benchmark_questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chroma-dir", type=Path, default=Path(".cache/advanced_chroma"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=2400)
    parser.add_argument("--methods", default="tfidf,dense,hybrid_rrf")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    selected = [q for q in json.loads(args.questions.read_text(encoding="utf-8")) if q["split"] == args.split]
    questions = [q for q in selected if q["answerable"]]

    wanted = set(args.methods.split(","))
    lexical = LexicalRetriever(chunks)
    dense = None
    if wanted & {"dense", "hybrid_rrf"}:
        dense = create_dense_retriever(chunks, args.chroma_dir, model_name=EMBEDDING_MODEL)
    registry = {
        "tfidf": lexical,
        "dense": dense,
        "hybrid_rrf": HybridRetriever(dense, lexical) if dense else None,
    }

    summaries, traces = [], {}
    for name in ["tfidf", "dense", "hybrid_rrf"]:
        if name not in wanted:
            continue
        summary, method_traces = evaluate(name, registry[name], questions, top_k=args.top_k, budget=args.context_budget)
        summaries.append(summary)
        traces[name] = method_traces

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "advanced_retrieval_summary.json").write_text(json.dumps({
        "split": args.split,
        "questions_total": len(selected),
        "answerable": len(questions),
        "unanswerable_excluded": len(selected) - len(questions),
        "top_k": args.top_k,
        "context_budget_chars": args.context_budget,
        "embedding_model": EMBEDDING_MODEL,
        "summaries": summaries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "advanced_retrieval_traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
