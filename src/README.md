# Retrieval code

`retrieval.py` implements three comparable methods:

- TF-IDF lexical retrieval, reusing the offline baseline.
- Dense retrieval backed by persistent Chroma and
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Hybrid retrieval using reciprocal-rank fusion (RRF), not unnormalized score addition.

The dense index is created/upserted with `create_dense_retriever(chunks, persist_dir)`.
The embedding model is downloaded on first use; the same model is used for indexing
and query encoding. User filters are applied to source fields or chunk metadata.

## Create the corpus

The repository's example fixture is not a benchmark. To create a larger corpus
from cited public material, run:

```powershell
python -m src.fetch_wikimedia --output data/wikimedia_chunks.json
```

The downloader saves article URL, revision ID, section, and CC BY-SA 4.0
attribution metadata. Each section is treated as a numbered page for the
assignment schema; this is a section ordinal, not a printed page number. The
corpus is English because relevant coverage is broader; the selected embedding
model is multilingual. Review the license and attribution before sharing or
submitting copied article text.

The local model with tool calling is not configured in this workspace yet.
`OllamaController` provides the adapter; install Ollama, pull a tool-capable
model (default `qwen2.5:3b`), start its local server, then construct the
controller and pass it with a retriever to `starter.agentic.run_agent`. The
adapter makes real HTTP calls and propagates connection/model errors.

## Retrieval benchmark

The curated benchmark is in `data/benchmark_questions.json` (15 answerable +
2 unanswerable dev cases; 5 answerable + 2 unanswerable test cases). Gold
evidence is validated against source and section before each run. Run dev first:

```powershell
python -m src.evaluate_retrieval --split dev --output results/retrieval-dev
```

Try context/top-k configurations on dev, then freeze the choice before test:

```powershell
python -m src.evaluate_retrieval --split dev --top-k 5 --context-budget 2400 --output results/retrieval-dev-k5
python -m src.evaluate_retrieval --split test --top-k 5 --context-budget 2400 --output results/retrieval-test
```

The runner compares TF-IDF, multilingual dense retrieval, and RRF hybrid at
the requested top-k and context budget; Evidence Hit@3 remains fixed at three.
It writes per-query traces and Hit@1, Evidence Hit@3, Context Evidence Rate,
and Hit@k. Local timings are machine dependent. The results directory and
Chroma index are ignored by Git.

## Run and audit the local agent

Install the Python dependencies, ensure the selected Hugging Face model is
cached locally, and run dev before test:

```powershell
python -m pip install -r requirements.txt
python -m src.run_agent_benchmark --split dev --output results/agent-dev
python -m src.run_agent_benchmark --split test --output results/agent-test
```

By default this uses the local `Qwen/Qwen2.5-0.5B-Instruct` Transformers model,
the multilingual embedding model, and hybrid RRF. Override `--model`,
`--provider`, `--retriever`, `--max-searches`, `--top-k`, or
`--context-budget` as needed. `agent_run.json` contains model/tool traces and
answers but no benchmark gold. Human-audit each answer and citation, then
aggregate only completed audits with `starter/agent_metrics.py`. Model load or
tool-call failures are recorded as errors, not counted as successful answers.

## Agent với Claude API

Đặt khóa vào file `.env` ở thư mục gốc (đã gitignore), hoặc biến môi trường:

```
ANTHROPIC_API_KEY=<khóa của bạn>
```

Chạy agent cơ bản bằng Claude (cấu hình đã chốt: hybrid RRF, top-5, 2400 ký tự, 1 lượt search):

```powershell
python -m src.run_agent_benchmark --split dev --provider anthropic --model claude-haiku-5-5 --retriever hybrid --top-k 5 --context-budget 2400 --max-searches 1 --max-new-tokens 512 --output results/basic-agent-dev-claude
python -m src.run_agent_benchmark --split test --provider anthropic --model claude-haiku-5-5 --retriever hybrid --top-k 5 --context-budget 2400 --max-searches 1 --max-new-tokens 512 --output results/basic-agent-test-claude
```

Adapter gọi Messages API trực tiếp qua `urllib`, không cần SDK; `disable_parallel_tool_use` được bật để mỗi lượt chỉ có một tool call. Lỗi HTTP chỉ báo mã trạng thái, không in nội dung phản hồi hay khóa.

## Thử thách nâng cao (tài liệu PDF 100 trang)

Đặt file PDF vào `data/source/` (thư mục này đã gitignore vì tài liệu có thông tin cá nhân). Chạy theo thứ tự:

```powershell
python -m pip install -r requirements.txt
python -m src.build_pdf_corpus "data/source/<ten_file>.pdf" --out data/advanced --source-id thesis --figure-notes data/advanced/figure_notes.json
python -m src.build_advanced_benchmark
python -m src.validate_advanced_benchmark
python -m src.evaluate_advanced_retrieval --split dev --top-k 5 --context-budget 2400 --output results/advanced-dev
python -m src.evaluate_advanced_retrieval --split test --top-k 5 --context-budget 2400 --methods tfidf --output results/advanced-test-k5-2400
python -m src.run_agent_benchmark --chunks data/advanced/chunks.json --questions data/advanced/benchmark_questions.json --split test --provider anthropic --model claude-haiku-5-5 --retriever tfidf --top-k 5 --context-budget 2400 --max-searches 1 --max-new-tokens 512 --output results/advanced-agent-test-claude
```

- `figure_notes.json` là ghi chú hình do người làm viết sau khi xem ảnh trang; không phải đầu ra của model. Cần kiểm tra lại với hình gốc.
- `build_advanced_benchmark.py` chứa câu hỏi và bằng chứng dev/test. `validate_advanced_benchmark.py` kiểm tra mọi bằng chứng là chuỗi nguyên văn nằm đúng trang và từ khóa của câu thiếu bằng chứng không xuất hiện trong tài liệu.
- Trang renders (`data/advanced/renders/`) và `results/` không commit.
