# Day 7 — Agentic RAG report

> Rename this file to `<MSSV>_<HoTen>.md` and add personal/course details before submission.

## Data and benchmark

- Corpus: 30 English Wikipedia articles on information retrieval, embeddings, ranking, NLP, and RAG; 578 chunks in the checked-in JSON export.
- Attribution and exact article revisions: [`WIKIMEDIA_ATTRIBUTION.txt`](../../data/WIKIMEDIA_ATTRIBUTION.txt). Text is CC BY-SA 4.0; each chunk retains the source URL and revision.
- `page` is the article section ordinal, not a printed page number.
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; dense store: persistent Chroma using cosine distance.
- Benchmark: [`benchmark_questions.json`](../../data/benchmark_questions.json), 15 answerable + 2 unanswerable dev questions and 5 answerable + 2 unanswerable test questions. Gold evidence is checked against source ID and section before retrieval runs. Retrieval metrics exclude unanswerable cases.
- Local agent model: `Qwen/Qwen2.5-0.5B-Instruct` via Transformers on CPU. The model and embedding checkpoints were already cached locally; inference is offline. Local service cost is $0.

## Retrieval experiments

All three methods use the same corpus and queries. `Evidence Hit@3` always evaluates the first three results; `Hit@k` is additionally reported when using top-5. Context budget is in characters. The context packer includes source headers and truncates long chunks rather than silently dropping them.

| Method / split | top-k / context | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Hit@k | Median query time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF / dev | 3 / 800 | 9/15 (60.0%) | 11/15 (73.3%) | 10/15 (66.7%) | 11/15 (73.3%) | 4.96 ms |
| Dense / dev | 3 / 800 | 6/15 (40.0%) | 12/15 (80.0%) | 8/15 (53.3%) | 12/15 (80.0%) | 40.22 ms |
| Hybrid RRF / dev | 3 / 800 | 11/15 (73.3%) | 13/15 (86.7%) | 12/15 (80.0%) | 13/15 (86.7%) | 43.50 ms |
| Hybrid RRF / dev, trial 1 | 3 / 1600 | 11/15 (73.3%) | 13/15 (86.7%) | 12/15 (80.0%) | 13/15 (86.7%) | 31.27 ms |
| Hybrid RRF / dev, selected | 5 / 2400 | 11/15 (73.3%) | 13/15 (86.7%) | 13/15 (86.7%) | 14/15 (93.3%) | 35.40 ms |
| TF-IDF / test | 5 / 2400 | 1/5 (20.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 3/5 (60.0%) | 2.82 ms |
| Dense / test | 5 / 2400 | 2/5 (40.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 4/5 (80.0%) | 32.54 ms |
| Hybrid RRF / test | 5 / 2400 | 2/5 (40.0%) | 4/5 (80.0%) | 4/5 (80.0%) | 4/5 (80.0%) | 26.50 ms |

Trial 1 tested the hypothesis that a larger context alone would improve the selected hybrid pipeline; it did not change dev Context Evidence Rate. Trial 2 increased both top-k and context budget; the hybrid retained its Hit@3, while Hit@5 and Context Evidence Rate rose to 14/15 and 13/15. Hybrid RRF was selected because it had the strongest dev Hit@1 and Evidence Hit@3 among the compared methods. The test run followed that selection.

`results/` is intentionally Git-ignored. Recreate dev, the two dev trials, and test with the commands in [`src/README.md`](../../src/README.md). Timings are local CPU measurements and vary with model/cache warm-up.

## Agent, audit, and error analysis

The local Transformers agent uses one initial search of the original question,
then answers from the top-five context; this small CPU-only model is not used
for iterative query rewriting. Its citation normalizer accepts an ID explicitly
returned by the model, or links the answer to a retrieved chunk only when at
least half of its non-stopword terms overlap. The latter is a candidate citation,
not proof of support; human audit is still required. Model inference is slow on
this CPU, so output length is capped at 32 new tokens.

Dev QA traces are being generated with the frozen configuration. Afterward,
review every test answer and citation against its cited chunk; do not treat
lexical citation linking as proof of entailment. Complete test grounded-answer
grades and the two abstention reviews in the agent review JSON before submission.

Initial dev retrieval error examples:

- `dev-03` asks about TF-IDF local/global weights. The corpus contains equivalent evidence in both the vector-space-model and TF-IDF articles; both are listed as valid gold alternatives to avoid penalizing a correct cross-article retrieval.
- `dev-06` asks what learning-to-rank constructs. The top-three ranking can favor navigation/related-article chunks over the article lead. Top-five improves the chance of retrieving the definition; inspect the saved trace before claiming that context supports an answer.

Do not claim final answer quality until the human audit and `starter/agent_metrics.py` aggregation are complete. No advanced 100-page challenge was attempted.
