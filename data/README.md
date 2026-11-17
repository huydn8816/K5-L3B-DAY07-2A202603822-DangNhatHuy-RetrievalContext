# Định dạng dữ liệu

`wikimedia_chunks.json` là corpus 30 bài nguồn tiếng Anh về retrieval, ranking,
embeddings và RAG (578 chunk tại lần tạo hiện tại). Tạo lại bằng:

```powershell
python -m src.fetch_wikimedia --output data/wikimedia_chunks.json
```

Mỗi chunk lưu URL, revision, tên section và thông tin CC BY-SA 4.0; xem
[`WIKIMEDIA_ATTRIBUTION.txt`](./WIKIMEDIA_ATTRIBUTION.txt) trước khi chia sẻ.
Trường `page` là số thứ tự section trong bài web, không phải số trang in.
`benchmark_questions.json` chứa 15 câu answerable và 2 câu unanswerable cho dev,
cùng 5 câu answerable và 2 câu unanswerable cho test. Gold evidence được kiểm
tra theo chunk/source/section; câu unanswerable cần được audit khi đọc trace.
Không dùng fixture `example_*.json` để tuyên bố kết quả benchmark.

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).
