"""Run the real local-model agent and save auditable per-question traces."""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from starter.agentic import run_agent
from starter.baseline import TfidfIndex, validate
from src.anthropic_controller import AnthropicController
from src.ollama_controller import OllamaController
from src.retrieval import HybridRetriever, LexicalRetriever, create_dense_retriever
from src.transformers_controller import TransformersController


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class InitialSearchController:
    """Prime the agent with the original query before asking the model to reason."""

    def __init__(self, controller: Any):
        self.controller = controller

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {"action": "search", "query": question}
        if isinstance(self.controller, TransformersController):
            return self.controller.answer(question, history)
        return self.controller.decide(question, history)


def _save(output: Path, configuration: dict[str, Any], traces: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "agent_run.json").write_text(
        json.dumps({"configuration": configuration, "traces": traces}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "agent_traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real local RAG agent on one benchmark split")
    parser.add_argument("--chunks", type=Path, default=Path("data/wikimedia_chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark_questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever", choices=("tfidf", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--provider", choices=("transformers", "ollama", "anthropic"), default="transformers")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chroma-dir", type=Path, default=Path(".cache/day7_chroma"))
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=2400)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.max_searches < 1 or args.top_k < 1 or args.context_budget < 1:
        raise ValueError("max-searches, top-k và context-budget phải lớn hơn 0")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [question for question in all_questions if question.get("split") == args.split]
    # gold_all (multi-hop) items are checked by validate_advanced_benchmark.py
    validate(chunks, [q for q in selected if "gold_all" not in q])
    if not selected:
        raise ValueError(f"không có câu hỏi trong split {args.split}")

    lexical = LexicalRetriever(chunks)
    if args.retriever == "tfidf":
        retriever = lexical
    elif args.retriever == "dense":
        retriever = create_dense_retriever(chunks, args.chroma_dir)
    else:
        retriever = HybridRetriever(create_dense_retriever(chunks, args.chroma_dir), lexical)

    if args.provider == "transformers":
        model_controller = TransformersController.from_pretrained(
            args.model, max_new_tokens=args.max_new_tokens, cpu_threads=args.cpu_threads,
        )
    elif args.provider == "anthropic":
        model_controller = AnthropicController(model=args.model, max_tokens=args.max_new_tokens)
    else:
        model_controller = OllamaController(model=args.model)
    controller = InitialSearchController(model_controller)

    configuration = {
        "split": args.split,
        "model": args.model,
        "provider": args.provider,
        "retriever": args.retriever,
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if args.retriever in ("dense", "hybrid") else None,
        "top_k": args.top_k,
        "max_searches": args.max_searches,
        "initial_search": "original question",
        "context_budget_chars": args.context_budget,
        "max_new_tokens": args.max_new_tokens if args.provider == "transformers" else None,
        "cpu_threads": args.cpu_threads if args.provider == "transformers" else None,
        "local_service_cost_usd": 0,
    }
    traces = []
    _save(args.output, configuration, traces)
    for question in selected:
        started = time.perf_counter()
        try:
            result = run_agent(
                question["question"],
                controller,
                retriever,
                filters=question.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.context_budget,
            )
            trace = {
                "query_id": question["id"],
                "split": args.split,
                "answerable": question.get("answerable", True),
                **result,
            }
        except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as error:
            trace = {
                "query_id": question["id"],
                "split": args.split,
                "answerable": question.get("answerable", True),
                "status": "error",
                "answer": None,
                "citations": [],
                "events": [],
                "error": f"{type(error).__name__}: {error}",
            }
        trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
        traces.append(trace)
        _save(args.output, configuration, traces)
        print(json.dumps({
            "query_id": trace["query_id"],
            "status": trace["status"],
            "elapsed_ms": round(trace["elapsed_ms"], 1),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
