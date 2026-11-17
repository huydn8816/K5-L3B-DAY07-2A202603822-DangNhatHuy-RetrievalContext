# Báo cáo Day 7

- **Họ và tên:** Đặng Nhật Huy
- **MSSV:** 2A202603822
- **Lớp:** L3B
- **Repo cá nhân:** https://github.com/huydn8816/K5-L3B-DAY07-2A202603822-DangNhatHuy-RetrievalContext

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** 30 bài Wikipedia tiếng Anh về information retrieval, embeddings, ranking, NLP và RAG; 578 chunk trong bản export JSON. Nội dung theo CC BY-SA 4.0; URL và revision từng bài xem tại [WIKIMEDIA_ATTRIBUTION.txt](../../data/WIKIMEDIA_ATTRIBUTION.txt). Trường `page` là số thứ tự section, không phải số trang in.
- **Vector store, embedding model, phiên bản và số chiều:** Chroma cục bộ (cosine distance); embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều). **Kết nối với vector store Day 6: chưa xác nhận.** Sinh viên cho biết Day 6 dùng cùng dữ liệu, nhưng index Day 6 đã bị xóa nên không thể đối chiếu trực tiếp. Index dùng cho Day 7 được dựng lại từ cùng `data/wikimedia_chunks.json` (578 chunk) và cùng embedding model; kết quả top-5 trùng hoàn toàn ở 24/24 truy vấn dev và test so với index ban đầu. Đây là tái lập của index Day 7, không phải store Day 6 gốc. Cần giảng viên xác nhận có chấp nhận cách này cho mục kết nối Day 6 hay không.
- **Agent model, cách chạy và phân chia dev/test:** agent chính là `claude-haiku-5-5` qua Claude API (chọn vì nhanh và rẻ; chưa so sánh với model khác trên dev). Thử nghiệm ban đầu với `Qwen/Qwen2.5-0.5B-Instruct` (Transformers, CPU, chi phí $0) chỉ có trace dev, không dùng để kết luận. Benchmark: [benchmark_questions.json](../../data/benchmark_questions.json) gồm dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** Câu có đáp án có gold evidence (source ID, section, đoạn text), được kiểm tra trước khi chạy. Câu thiếu evidence được loại khỏi mẫu số retrieval metrics. Chấm agent theo thang 0/1/2 (xem mục Agent).

## Retrieval và thử nghiệm

Tất cả phương pháp dùng cùng corpus và cùng câu hỏi. Evidence Hit@3 luôn đo top-3; Hit@k chỉ báo khi top-k = 5. Context budget tính bằng ký tự.

| Cấu hình | Phương pháp, top-k, context | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (median) / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, 3 / 800 | Lexical baseline | dev | 9/15 (60.0%) | 11/15 (73.3%) | 10/15 (66.7%) | 4.96 ms / $0 |
| Phương pháp 2 | Dense, 3 / 800 | Ngữ nghĩa đa ngôn ngữ có thể tìm được câu diễn đạt khác | dev | 6/15 (40.0%) | 12/15 (80.0%) | 8/15 (53.3%) | 40.22 ms / $0 |
| Phương pháp 3 | Hybrid RRF, 3 / 800 | Kết hợp TF-IDF và dense bằng RRF | dev | 11/15 (73.3%) | 13/15 (86.7%) | 12/15 (80.0%) | 43.50 ms / $0 |
| Cải thiện 1 | Hybrid RRF, 3 / 1600 | Tăng context có thể đủ để cải thiện | dev | 11/15 (73.3%) | 13/15 (86.7%) | 12/15 (80.0%) | 31.27 ms / $0 |
| Cải thiện 2 (đã chọn) | Hybrid RRF, 5 / 2400 | Tăng top-k và context để tìm đủ evidence | dev | 11/15 (73.3%) | 13/15 (86.7%) | 13/15 (86.7%); Hit@5 14/15 (93.3%) | 35.40 ms / $0 |
| Cấu hình đã chọn | Hybrid RRF, 5 / 2400 | Chốt trên dev trước khi chạy test | test | 2/5 (40.0%) | 4/5 (80.0%) | 4/5 (80.0%); Hit@5 4/5 (80.0%) | 26.50 ms / $0 |
| Baseline | TF-IDF, 5 / 2400 | Đối chiếu trên test | test | 1/5 (20.0%) | 2/5 (40.0%) | 2/5 (40.0%); Hit@5 3/5 (60.0%) | 2.82 ms / $0 |

Ghi chú:
- Cải thiện 1 thử giả thuyết "chỉ tăng context"; không làm thay đổi Context Evidence Rate trên dev.
- Cải thiện 2 tăng cả top-k và context; Evidence Hit@3 giữ nguyên, Hit@5 và Context Evidence Rate tăng.
- Hybrid RRF được chọn vì có Hit@1 và Evidence Hit@3 cao nhất trên dev. Test chạy sau khi đã chốt cấu hình.
- Dense và TF-IDF trên test cũng được đo ở top-5/2400 để so sánh cùng điều kiện; số liệu dense test: Hit@1 2/5, Evidence Hit@3 3/5.

**Hai lỗi dev:**
1. `dev-03` (TF-IDF local/global weights): corpus có evidence tương đương ở cả bài vector space model và TF-IDF. Cả hai được đưa vào danh sách gold hợp lệ để không phạt việc tìm đúng ở bài khác.
2. `dev-06` (learning-to-rank): top-3 thường ưu tiên chunk điều hướng/bài liên quan hơn phần mở đầu định nghĩa. Top-5 tăng cơ hội lấy được định nghĩa.

Đường dẫn kết quả: `results/retrieval-dev*/` và `results/retrieval-test/` (thư mục này bị git-ignore, xem [src/README.md](../../src/README.md) để tái tạo).

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context, xử lý lỗi:** Một lượt search với câu hỏi gốc (hybrid RRF, top-5, 2400 ký tự), sau đó trả lời từ context. Không viết lại query. Tối đa 32 token mới. Citation chấp nhận ID model trả về, hoặc gắn với chunk đã retrieve khi ≥50% từ không phải stopword trùng khớp; đây chỉ là citation ứng viên, cần kiểm tra thủ công. Lỗi model/tool được ghi là lỗi, không tính là câu trả lời thành công.
- **Trạng thái:** Đã chạy và chấm agent Claude trên dev (17 câu, `results/basic-agent-dev-claude/`) và test (7 câu, `results/basic-agent-test-claude/`). Trace Qwen 0.5B trên dev (`results/agent-dev/`) vẫn chưa chấm và không dùng để kết luận.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent hybrid 5/2400 / dev (Claude haiku-5-5) | 17/17 gọi search | 1 | 15 câu có đáp án: 14 câu 2 điểm, 1 câu 0 điểm (trung bình 1.87/2) | 25/25 citation nằm trong context | 14/14 câu trả lời có citation | 2/2 | 0 |
| Agent hybrid 5/2400 / dev (Qwen 0.5B, chạy local, chưa chấm) | — | 1 | — | — | — | — | — |
| Agent hybrid 5/2400 / test (Claude haiku-5-5) | 7/7 gọi search | 1 | 5 câu có đáp án: 2/2/2/2/2 (trung bình 2.0/2) | 5/5 citation nằm trong context | 5/5 câu trả lời có citation | 2/2 | 0 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| test-01 | Có đáp án | 1 search, gold trong context | Trả lời đúng; citation `wiki-4059023-p1-c1` | 2: đúng, đủ, citation hỗ trợ |
| test-02 | Có đáp án | 1 search, gold trong context | Trả lời đúng cả hai phần; citation `wiki-43561218-p1-c1` | 2 |
| test-03 | Có đáp án | 1 search, gold trong context | Trả lời đúng; citation `wiki-14343887-p1-c1` | 2 |
| test-04 | Có đáp án | 1 search, gold trong context | Trả lời đúng lý do chuẩn hóa; citation p1, p5 | 2 |
| test-05 | Có đáp án | 1 search, gold (p3) không trong context | Trả lời đúng "hyperlink"; citation `wiki-26334893-p2-c1` hỗ trợ đúng ý, không phải trang gold | 2: chấp nhận vì citation hỗ trợ đúng ý (alternative) |
| test-06 | Thiếu bằng chứng | 1 search | Từ chối đúng | 2 |
| test-07 | Thiếu bằng chứng | 1 search | Từ chối đúng | 2 |
| dev-03 | Có đáp án (lỗi) | 1 search, một trong hai đoạn gold có trong context | Từ chối "không đủ bằng chứng" | 0: lỗi sinh câu trả lời, không phải lỗi retrieval |

- Đường dẫn trace: `results/agent-dev/agent_run.json`, `results/agent-dev/agent_traces.json`.
- Lệnh chạy Claude (dev, test): xem [src/README.md](../../src/README.md#agent-với-claude-api). Trace Qwen: `python -m src.run_agent_benchmark --split dev --output results/agent-dev`.
- Kết quả dev: 14/15 câu có đáp án đạt, 1 câu sai (dev-03, agent từ chối dù một trong hai đoạn gold đã có trong context), 2/2 câu thiếu bằng chứng đúng. Test: 7/7.
- Kết quả trước/sau: **chưa đo lại lỗi dev sau khi thay đổi** (chưa thử giả thuyết sửa dev-03 hoặc dev-06 rồi chạy lại). Đây là phần còn thiếu theo BENCHMARK.md. Giới hạn: agent chỉ có 1 lượt search, test cơ bản chỉ 7 câu nên chưa đủ để kết luận; nhãn được chấm bằng cách đối chiếu chunk trích dẫn và cần người nộp kiểm tra lại; chi phí API chưa đo được vì trace không ghi usage.
- Filter và query: không dùng metadata filter. Query đưa vào retrieval là câu hỏi gốc, không viết lại và không mở rộng. Gold không được đưa vào agent hay retrieval.

## Thử thách nâng cao (nếu thực hiện)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** đồ án tốt nghiệp "Xây dựng mô hình TimesNet dự báo nhu cầu sản phẩm trong quản trị chuỗi cung ứng" (ĐH Bách Khoa Hà Nội, 07/2024), 117 trang PDF, file `data/source/Sample.pdf`. Đây là đồ án của chính tác giả, làm ở trường khác (trang bìa ghi MSSV cũ 20200271; MSSV hiện tại 2A202603822). File chứa thông tin cá nhân, nên chỉ nộp nguồn và cách lấy file, không commit PDF nếu không cần.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Text: lấy bằng `pypdf` theo từng trang, khoảng trắng được chuẩn hóa, chia chunk 700 ký tự (overlap 100). Tổng 336 chunk, trong đó có 44 dòng chú thích "Bảng/Hình" làm chunk riêng.
  - Bảng: lấy từ text layer; đã kiểm tra các số dùng trong benchmark với trang gốc.
  - Hình: 7 ghi chú mô tả do người làm viết sau khi xem ảnh trang (Hình 3.1, 3.2, 3.5, 3.6, 3.10, 3.11, 3.12), đánh dấu `figure_note_manual`. Không dùng vision model hay OCR.
  - Chưa index nội dung hình: các sơ đồ kiến trúc Inception (Hình 2.10–2.18), 2.1–2.9, 3.3, 3.4, 3.7–3.9 và các biểu đồ còn lại. Trang có hình: 33 trong 117 trang.
  - Ảnh render 117 trang: `data/advanced/renders/` (dùng để kiểm tra).
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - Benchmark: `data/advanced/benchmark_questions.json` (tạo bởi `src/build_advanced_benchmark.py`). Dev 18 câu (15 có đáp án, 3 thiếu bằng chứng). Test 30 câu (24 có đáp án: 6 bảng, 5 ảnh, 5 biểu đồ, 4 multi-hop, 4 text; 6 thiếu bằng chứng).
  - Schema: `gold` (một trong các đoạn hợp lệ), `gold_all` (multi-hop, mọi đoạn bắt buộc), `page` là số trang PDF (số trang in lệch 1). Kiểm tra bằng `src/validate_advanced_benchmark.py`: mọi bằng chứng là chuỗi nguyên văn trong chunk đúng trang; các từ khóa của câu thiếu bằng chứng không xuất hiện trong PDF.
  - Tiêu chí chọn cấu hình (chốt trước khi chạy test): context evidence rate cao nhất trên dev, hòa thì Hit@3.
  - Kết quả dev (context evidence rate, 15 câu có đáp án): TF-IDF top-3/800 11/15; **TF-IDF top-5/2400 12/15 (chọn)**; hybrid RRF top-5/2400 11/15; dense top-3/800 7/15.
  - Model agent: Claude `claude-haiku-5-5` (xem kết quả bên dưới).

| Nhóm test | Đạt / tổng (agent Claude) | Context evidence rate (TF-IDF top-5/2400) | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 1/6 | 5/6 | test-01, 02, 04, 06: bằng chứng đã có trong context nhưng agent vẫn từ chối (lỗi sinh câu trả lời). test-03: đáp án nằm ở hai chunk, không đủ trong context (lỗi truy xuất) |
| Ảnh, sơ đồ | 3/5 | 3/5 | test-08, 09 (Hình 3.6): ghi chú không vào context (lỗi truy xuất) |
| Biểu đồ | 3/5 | 5/5 | test-12, 14: bằng chứng đã có trong context nhưng agent vẫn từ chối (lỗi sinh câu trả lời) |
| Multi-hop | 0/4 | 0/4 | Context thiếu ít nhất một trong hai đoạn ở cả 4 câu; agent chỉ có 1 lượt search |
| Text | 3/4 | 3/4 | test-23: đoạn trang 65 không vào context (lỗi truy xuất) |
| Thiếu bằng chứng | 6/6 | — | Từ chối đúng cả 6 |
| **Toàn bộ** | **16/30 (53.3%)** | 16/24 (có đáp án) | Cần ≥27/30 (90%): không đạt |

- **Trace, nhãn và evidence từng câu:** `results/advanced-test-k5-2400/`, `results/advanced-dev/`, `results/advanced-test-k3-800/` (không commit, do `results/` bị git-ignore). Lệnh: `python -m src.evaluate_advanced_retrieval --split test --top-k 5 --context-budget 2400 --methods tfidf`.
- **Kết quả test (retrieval, 24 câu có đáp án):**

| Cấu hình | Hit@1 (câu đơn) | Hit@3 | Hit@5 | Context evidence rate |
|---|---|---|---|---|
| TF-IDF top-3 / 800 (baseline) | 10/20 | 16/24 | 16/24 | 15/24 (62.5%) |
| **TF-IDF top-5 / 2400 (chọn)** | 10/20 | 16/24 | 17/24 | **16/24 (66.7%)** |
| Dense top-5 / 2400 | 6/20 | 10/24 | 13/24 | 13/24 (54.2%) |
| Hybrid RRF top-5 / 2400 | 8/20 | 12/24 | 17/24 | 14/24 (58.3%) |

- **Agent (Claude `claude-haiku-5-5`, cấu hình đã chốt, 1 lượt search):**
  - Dev: 18 câu, 0 lỗi, đạt 8/18 (5/15 câu có đáp án, 3/3 từ chối đúng). Điểm trung bình grounded 0.733/2.
  - Test: 30 câu, 0 lỗi, đạt 16/30 (53.3%): 10/24 câu có đáp án, 6/6 câu thiếu bằng chứng. Điểm trung bình grounded 0.875/2 (phân bố 0: 13, 1: 1, 2: 10). 13/24 câu có đáp án bị từ chối.
  - Lỗi chính: 6 câu có đáp án đã có bằng chứng trong context nhưng agent vẫn từ chối (test-01, 02, 04, 06, 12, 14), còn lại là lỗi truy xuất.
  - Citation: 20/20 nằm trong context mà agent đã nhận; 11/11 câu trả lời có citation.
  - Nhãn từng câu: `data/advanced/agent_labels_claude.json`. Nhãn được chấm bằng cách đối chiếu đoạn chunk được trích dẫn với bằng chứng gốc. Nhãn cần người nộp kiểm tra lại trước khi nộp; ghi chú hình là ghi chú thủ công, chưa đối chiếu hết với hình gốc.
  - Chi phí và thời gian: API không ghi usage trong trace, nên chưa báo cáo được chi phí chính xác.
- **Kết luận và giới hạn:** Với retrieval một lượt, bonus không đạt. Trần lý thuyết là 22/30 (73%) và kết quả thực là 16/30 (53.3%), đều dưới ngưỡng 90% (27/30). Cần cải thiện đồng thời: (a) multi-hop (0/4) bằng agent tìm nhiều lượt, (b) ảnh và sơ đồ bằng mô tả hình hoặc vision model, (c) tăng tỷ lệ trả lời câu có đáp án. Tất cả giá trị biểu đồ là xấp xỉ đọc từ hình.
