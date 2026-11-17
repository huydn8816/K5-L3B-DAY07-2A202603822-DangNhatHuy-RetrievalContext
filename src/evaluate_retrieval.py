"""Compare lexical, dense, and RRF retrieval on one benchmark split."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from starter.baseline import TfidfIndex, hit, pack_context, validate
from src.retrieval import HybridRetriever, LexicalRetriever, create_dense_retriever


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
CONTEXT_BUDGET = 800


def _rate(n: int, total: int) -> dict[str, Any]:
    return {"n": n, "N": total, "rate": n / total if total else None}


def evaluate_method(name: str, retriever: Any, questions: list[dict[str, Any]], *,
                    index_seconds: float, top_k: int = TOP_K,
                    budget: int = CONTEXT_BUDGET) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    for question in questions:
        started = time.perf_counter()
        results = retriever.search(question["question"], filters=question.get("filter"), top_k=top_k)
        if len(results) > top_k or len({chunk["id"] for chunk in results}) != len(results):
            raise ValueError(f"{name}: retriever returned too many or duplicate chunk IDs")
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**chunk, "hit": hit(chunk, question["gold"])} for chunk in results]
        context_hit = any(chunk["hit"] and chunk["id"] in context_ids for chunk in marked)
        traces.append({
            "query_id": question["id"],
            "split": question["split"],
            "question": question["question"],
            "answerable": True,
            "strategy": name,
            "top_k": top_k,
            "budget_chars": budget,
            "gold": question["gold"],
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "evidence_hit3": any(chunk["hit"] for chunk in marked[:3]),
            "hitk": any(chunk["hit"] for chunk in marked),
            "hitk_cutoff": top_k,
            "context_hit": context_hit,
        })

    total = len(traces)
    if not total:
        raise ValueError("không có câu hỏi có đáp án để đánh giá retrieval")
    summary = {
        "strategy": name,
        "n": total,
        "hit1": _rate(sum(row["hit1"] for row in traces), total),
        "evidence_hit3": _rate(sum(row["evidence_hit3"] for row in traces), total),
        "context_evidence_rate": _rate(sum(row["context_hit"] for row in traces), total),
        "hitk": _rate(sum(row["hitk"] for row in traces), total),
        "hitk_cutoff": top_k,
        "avg_context_chars": statistics.mean(row["context_chars"] for row in traces),
        "median_query_ms": statistics.median(row["elapsed_ms"] for row in traces),
        "index_seconds": index_seconds,
        "local_service_cost_usd": 0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TF-IDF, dense, and hybrid RRF retrieval")
    parser.add_argument("--chunks", type=Path, default=Path("data/wikimedia_chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark_questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chroma-dir", type=Path, default=Path(".cache/day7_chroma"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    args = parser.parse_args()
    if args.top_k < 3 or args.context_budget < 1:
        raise ValueError("top-k phải >= 3 để so sánh Evidence Hit@3; context budget phải > 0")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [question for question in all_questions if question.get("split") == args.split]
    validate(chunks, selected)
    questions = [question for question in selected if question.get("answerable", True)]
    if not questions:
        raise ValueError(f"split {args.split} không có câu hỏi có đáp án")

    started = time.perf_counter()
    lexical = LexicalRetriever(chunks)
    lexical_seconds = time.perf_counter() - started

    dense_started = time.perf_counter()
    dense = create_dense_retriever(chunks, args.chroma_dir, model_name=EMBEDDING_MODEL)
    dense_seconds = time.perf_counter() - dense_started

    methods = [
        ("tfidf", lexical, lexical_seconds),
        ("dense", dense, dense_seconds),
        ("hybrid_rrf", HybridRetriever(dense, lexical), dense_seconds),
    ]
    summaries, traces = [], {}
    for name, retriever, build_seconds in methods:
        summary, method_traces = evaluate_method(
            name, retriever, questions, index_seconds=build_seconds,
            top_k=args.top_k, budget=args.context_budget,
        )
        summaries.append(summary)
        traces[name] = method_traces

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "retrieval_summary.json").write_text(
        json.dumps({
            "split": args.split,
            "questions": len(selected),
            "answerable_questions": len(questions),
            "top_k": args.top_k,
            "evidence_hit3_cutoff": 3,
            "context_budget_chars": args.context_budget,
            "embedding_model": EMBEDDING_MODEL,
            "summaries": summaries,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "retrieval_traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
