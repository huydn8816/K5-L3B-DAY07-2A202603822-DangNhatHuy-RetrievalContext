# Nộp bài Day 7

> Trạng thái cập nhật 2026-11-17: phần retrieval và benchmark đã có; agent trên test, chấm thủ công và báo cáo cá nhân còn dang dở. Xem [Việc còn lại](#việc-còn-lại-trước-khi-nộp).

## Tạo repo cá nhân

Đặt tên theo lớp, ví dụ `K5-L3A-DAY07-MSSV-HoTen-RetrievalContext` hoặc `K5-L3B-DAY07-MSSV-HoTen-RetrievalContext`. Tên không dấu, không khoảng trắng. Fork repo đề bài hoặc clone rồi đổi remote:

```bash
git clone https://github.com/VinUni-AI20k/K5-L3-Day07-Retrieval-Context-Engineering.git <TEN_REPO>
cd <TEN_REPO>
git remote rename origin upstream
git remote add origin <URL_REPO_CA_NHAN>
git push -u origin main
```

Thay phần trong `< >` bằng tên và địa chỉ thật. Nộp link theo hướng dẫn giảng viên; không push bài vào repo đề bài.

## Danh sách bài nộp

- [x] Code agent, ít nhất 3 phương pháp retrieval (gồm dense), test và hướng dẫn chạy.
  - Có TF-IDF, dense (multilingual MiniLM + Chroma) và hybrid RRF trong [src/retrieval.py](src/retrieval.py); 36 unit test trong `tests/` chạy đạt.
  - Hướng dẫn chạy: [src/README.md](src/README.md).
- [x] Embedding model, vector store, cách kết nối và dữ liệu được phép chia sẻ hoặc cách cấp quyền/tái tạo.
  - Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; vector store: Chroma cục bộ (không commit, tái tạo theo [src/README.md](src/README.md)).
  - Corpus: 30 bài Wikipedia (CC BY-SA 4.0), ghi nguồn tại [data/WIKIMEDIA_ATTRIBUTION.txt](data/WIKIMEDIA_ATTRIBUTION.txt); tái tạo bằng `src.fetch_wikimedia`.
  - Cần xác nhận lại cách kết nối với vector store của Day 6 (rubric yêu cầu) và ghi rõ trong báo cáo.
- [x] Benchmark gồm ít nhất 20 câu có evidence và 4 câu thiếu evidence, đã chia dev/test.
  - [data/benchmark_questions.json](data/benchmark_questions.json): dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence (tổng 20 + 4).
- [ ] Kết quả từng query và tool call; context, câu trả lời, citation, lỗi, thời gian và chi phí.
  - Đã có: trace retrieval trong `results/retrieval-*` và trace agent dev trong `results/agent-dev/`.
  - Chưa có: trace agent trên test; `results/` đang bị git-ignore nên trace chưa nằm trong repo nộp.
- [ ] Ít nhất 2 cấu hình cải thiện; bảng benchmark, nhãn, phân tích lỗi và lý do chọn theo [BENCHMARK.md](BENCHMARK.md).
  - Đã có cho retrieval: TF-IDF → dense → hybrid RRF, rồi tăng top-k/context (chọn hybrid, top-5, 2400 ký tự); bảng và 2 lỗi dev trong [báo cáo retrieval](report/individual/Day07_Retrieval_Results.md).
  - Chưa có: nhãn agent (grounded answer, citation, abstention) cho dev và test.
- [ ] Sao chép và điền [báo cáo cá nhân](report/individual/2A202603822_DangNhatHuy.md), đổi tên theo MSSV và họ tên.
  - Mới có bản nháp retrieval; chưa đổi tên, chưa điền thông tin cá nhân và phần agent.
- Repo cá nhân: https://github.com/huydn8816/K5-L3B-DAY07-2A202603822-DangNhatHuy-RetrievalContext
- [ ] Người chấm có quyền xem repo; không nộp dữ liệu mật, `.env`, API key hoặc token.
  - Cần kiểm tra trước khi push; hiện nhiều file vẫn chưa commit.

## Nếu làm thử thách nâng cao

Đã làm phần tài liệu, benchmark và retrieval; agent và chấm thủ công chưa làm. Xem [ADVANCED.md](ADVANCED.md).

- [x] Nguồn và quyền truy cập tài liệu từ 100 trang; thống kê, cấu hình index và phần chưa xử lý.
  - Tài liệu: đồ án tốt nghiệp 117 trang, `data/source/Sample.pdf`. Đây là đồ án của bạn ở trường khác (MSSV cũ 20200271 trên bìa).
  - Phạm vi index và phần hình chưa xử lý: xem [báo cáo](report/individual/2A202603822_DangNhatHuy.md).
- [x] Benchmark dev/test, đáp án và evidence theo từng trang/vùng, gồm bảng, ảnh và biểu đồ.
  - [data/advanced/benchmark_questions.json](data/advanced/benchmark_questions.json): dev 18, test 30; đã kiểm tra bằng [validate_advanced_benchmark.py](src/validate_advanced_benchmark.py).
- [ ] Trace, nhãn và lý do cho toàn bộ test; tỷ lệ đạt chung và theo nhóm, lỗi, thời gian và chi phí.
  - Mới có trace retrieval. Chưa chạy agent và chưa chấm từng câu test.
- [ ] Cách chạy tái lập và bằng chứng đạt ít nhất 90% theo [ADVANCED.md](ADVANCED.md).
  - Cách chạy retrieval đã có. Bằng chứng đạt 90% chưa có: trần hiện tại là 22/30 (73%).

## Việc còn lại trước khi nộp

1. **Kết nối vector store Day 6** (hoặc ghi rõ lý do không dùng) trong báo cáo cá nhân.
2. **Xác nhận repo đích và tên repo** theo mẫu `K5-L3B-DAY07-MSSV-HoTen-RetrievalContext`. Hiện remote khác URL đã đưa.
3. **Đo lại lỗi dev** sau khi thay đổi cấu hình (BENCHMARK.md yêu cầu giả thuyết, kết quả trước/sau).
4. **Kiểm tra nhãn chấm** (nhãn cơ bản và nâng cao, trong `data/*labels*.json`) trước khi nộp.
5. **Đưa trace vào nơi người chấm xem được:** `results/` đang bị git-ignore.
6. **Không commit** `.env`, `data/source/` (PDF cá nhân), `data/advanced/pages.json`, `data/advanced/chunks.json` (toàn văn đồ án).
7. Push lên nhánh của repo cá nhân sau khi xác nhận repo đích.
