"""Write the advanced-challenge benchmark (dev/test) for the thesis PDF.

Gold evidence strings are checked against chunk text by validate_advanced_benchmark.py.
Dev is for choosing the configuration; test is only for the frozen configuration.
"""
import json
from pathlib import Path

S = "thesis"
OUT = Path("data/advanced/benchmark_questions.json")


def g(page, ev):
    return {"source_id": S, "page": page, "evidence": ev}


Q = []


def add(id_, split, group, question, expected, gold=None, all_of=None):
    item = {"id": id_, "split": split, "group": group, "answerable": True,
            "question": question, "expected_answer": expected}
    if gold:
        item["gold"] = gold
    if all_of:
        item["gold_all"] = all_of
    Q.append(item)


def absent(id_, split, question, reason, terms):
    Q.append({"id": id_, "split": split, "group": "unanswerable", "answerable": False,
              "question": question, "expected_behavior": "search_and_abstain",
              "absence_reason": reason, "absence_terms": terms})


# ---------- DEV: 15 answerable (table 3, image 3, chart 3, multihop 2, text 4) + 3 unanswerable ----------
add("dev-01", "dev", "table", "Trong Bảng 3.5, kịch bản 1 có bao nhiêu mẫu ở tập huấn luyện (Train)?", "770",
    [g(76, "(770, 165, 165)")])
add("dev-02", "dev", "table", "Với mô hình TimesNet + InceptionBlock V2 và độ dài dự báo 100, số tham số là bao nhiêu?", "12775",
    [g(83, "100 12775 0.0487")])
add("dev-03", "dev", "table", "Trong Bảng 3.8 (kịch bản 1), MSE trung bình của LED2 với TN+IB V2 là bao nhiêu?", "3.42e4",
    [g(85, "3.5e4 3.42e4")])
add("dev-04", "dev", "image", "Trong Hình 3.11, mỗi biểu đồ con có bao nhiêu đường biểu diễn?", "2 (Prediction và True)",
    [g(86, "Đường đỏ là Prediction (dự báo), đường xanh là True (thực tế)")])
add("dev-05", "dev", "image", "Hình 3.11 có bao nhiêu biểu đồ con?", "6",
    [g(86, "Hình 3.11 gồm sáu biểu đồ con (a) đến (f)")])
add("dev-06", "dev", "image", "Trong Hình 3.6, tập Validation của kịch bản 2 kéo dài từ ngày nào đến ngày nào?",
    "Từ 08-02-2023 đến 02-06-2023", [g(77, "Validation 08-02-2023 đến 02-06-2023")])
add("dev-07", "dev", "chart", "Trong Hình 3.10, sản phẩm LED1 có MAPE thấp nhất tại mọi giá trị K không?", "Có",
    [g(82, "LED1 thấp nhất và LED2 cao nhất tại mọi giá trị K")])
add("dev-08", "dev", "chart", "Trong Hình 3.12, trục hoành biểu diễn đại lượng nào?", "Epochs",
    [g(87, "MSE theo Epochs cho V1 đến V6")])
add("dev-09", "dev", "chart", "Trong Hình 3.2(c), số lượng bán ra của LED3 tập trung ở vùng giá trị nào?",
    "Vùng giá trị nhỏ gần 0", [g(68, "Biểu đồ (c) LED3 tập trung ở vùng giá trị nhỏ gần 0")])
add("dev-10", "dev", "multihop", "Kịch bản 2 có bao nhiêu mẫu ở tập Test, và tập Test bắt đầu từ ngày nào?",
    "110 mẫu; bắt đầu từ 03-06-2023",
    all_of=[g(76, "(880, 110, 110)"), g(77, "Test 03-06-2023 đến 31-12-2023")])
add("dev-11", "dev", "multihop",
    "Sản phẩm LED3 thuộc loại sản phẩm nào, và biểu đồ mật độ của LED3 tập trung ở vùng giá trị nào?",
    "Bộ đèn LED M26 (loại Bộ đèn M16, M26, M36); vùng giá trị nhỏ gần 0",
    all_of=[g(65, "LED3 Bộ đèn LED M26 1200/40W 6500K Bộ đèn M16, M26, M36"),
            g(68, "Biểu đồ (c) LED3 tập trung ở vùng giá trị nhỏ gần 0")])
add("dev-12", "dev", "text", "Bảng 3.7 đo những yếu tố hiệu suất nào của các phiên bản Inception Block?",
    "Số lượng tham số, bộ nhớ tham số và thời gian chạy",
    [g(82, "đối với 3 yếu tố bao gồm: số lượng tham số, bộ nhớ tham số và thời gian chạy")])
add("dev-13", "dev", "text", "Tác giả chuẩn hóa dữ liệu bằng phương pháp nào?", "Chuẩn hóa Z-score",
    [g(65, "theo phương pháp chuẩn hóa Z-score")])
add("dev-14", "dev", "text", "Theo phân tích, LED4 không được ưu chuộng trên thị trường trong giai đoạn nào?",
    "Từ năm 2021 đến 2023", [g(67, "không được ưu chuộng trên thị trường từ năm 2021 đến 2023")])
add("dev-15", "dev", "text", "Số khối TimesBlock được sử dụng trong mô hình TimesNet là bao nhiêu?", "5",
    [g(76, "Số lượng khối TimesBlock sử dụng trong mô hình TimesNet là 5")])
absent("dev-16", "dev", "Tài liệu có báo cáo doanh thu năm 2024 của sản phẩm LED1 không?",
       "Tài liệu không có số liệu doanh thu.", ["doanh thu"])
absent("dev-17", "dev", "Mô hình có dự báo nhu cầu cho sản phẩm LED6 không?",
       "Tài liệu chỉ có LED1 đến LED5.", ["LED6"])
absent("dev-18", "dev", "Thời gian huấn luyện của TimesNet + InceptionBlock V1 là bao nhiêu giây?",
       "Tài liệu không báo cáo thời gian huấn luyện theo giây.", ["giây"])

# ---------- TEST: 24 answerable (table 6, image 5, chart 5, multihop 4, text 4) + 6 unanswerable ----------
add("test-01", "test", "table", "Trong Bảng 3.5, tập Validation của kịch bản 1 có bao nhiêu mẫu?", "165",
    [g(76, "(770, 165, 165)")])
add("test-02", "test", "table", "Trong Bảng 3.7, mô hình TimesNet + InceptionBlock V1 với độ dài dự báo 50 có bao nhiêu tham số?",
    "7405", [g(83, "50 7405 0.0289")])
add("test-03", "test", "table", "Trong Bảng 3.8 (kịch bản 1), MAE trung bình của LED1 với TN+IB V3 là bao nhiêu?", "158.14",
    [g(85, "MAE 128.25 186.51 158.14")])
add("test-04", "test", "table", "Theo Bảng 3.1, sản phẩm LED2 thuộc loại sản phẩm nào?", "Bộ đèn M16, M26, M36",
    [g(65, "LED2 Bộ đèn LED M36 1200/40W 6500K Bộ đèn M16, M26, M36")])
add("test-05", "test", "table", "Trong Bảng 3.6, batch size và số epochs là bao nhiêu?", "Batch size 32, 50 epochs",
    [g(76, "MSE 32 50")])
add("test-06", "test", "table", "Trong Bảng 3.8 (kịch bản 1), MAPE trung bình của LED2 với TN+IB V6 là bao nhiêu?", "14.50",
    [g(85, "MAPE 10.20 12.42 11.31 13.51 15.50 14.50 12.33 14.10 13.21")])
add("test-07", "test", "image", "Trong Hình 3.5, có bao nhiêu khối TimesBlock được vẽ nối tiếp?", "3",
    [g(73, "Hình 3.5 là sơ đồ mô tả thuật toán, có ba khối TimesBlock")])
add("test-08", "test", "image", "Trong Hình 3.6, kịch bản 1 chia dữ liệu theo tỷ lệ nào?", "70%:15%:15%",
    [g(77, "Kịch bản 1 (70%:15%:15%)")])
add("test-09", "test", "image", "Trong Hình 3.6, tập Test của kịch bản 2 kết thúc vào ngày nào?", "31-12-2023",
    [g(77, "Test 03-06-2023 đến 31-12-2023")])
add("test-10", "test", "image", "Trong Hình 3.11, đường màu đỏ biểu diễn đại lượng nào?", "Dự báo (Prediction)",
    [g(86, "Đường đỏ là Prediction (dự báo)")])
add("test-11", "test", "image", "Trong Hình 3.1(a), các điểm ngoại lai của biểu đồ hộp nằm ở phía nào của hộp?",
    "Phía trên hộp", [g(67, "các điểm ngoại lai nằm phía trên hộp")])
add("test-12", "test", "chart", "Trong Hình 3.10, ở K = 5 sản phẩm nào có MAPE thấp nhất?", "LED1",
    [g(82, "LED1 thấp nhất và LED2 cao nhất tại mọi giá trị K")])
add("test-13", "test", "chart", "Trong Hình 3.10, MAPE của LED2 có giảm khi K tăng từ 1 đến 5 không?",
    "Có, giảm khi K tăng", [g(82, "Cả ba đường đều giảm khi K tăng")])
add("test-14", "test", "chart", "Trong Hình 3.2(d), số lượng bán ra của LED4 tập trung ở khoảng giá trị nào?", "Gần 0",
    [g(68, "Biểu đồ (d) LED4 gần như toàn bộ giá trị tập trung gần 0")])
add("test-15", "test", "chart", "Trong Hình 3.1(b), các điểm bất thường được đánh dấu bằng màu gì?", "Đỏ",
    [g(67, "các điểm bất thường được đánh dấu bằng chấm màu đỏ")])
add("test-16", "test", "chart", "Trong Hình 3.12, đường biểu diễn tập Train có màu gì?", "Xanh dương",
    [g(87, "Đường xanh dương là Train")])
add("test-17", "test", "multihop", "Kịch bản 1 có bao nhiêu mẫu ở tập Train và được chia theo tỷ lệ nào?",
    "770 mẫu; 70%:15%:15%",
    all_of=[g(76, "(770, 165, 165)"), g(77, "Kịch bản 1 (70%:15%:15%)")])
add("test-18", "test", "multihop", "Giá trị K được chọn là bao nhiêu, và LED1 có MAPE xấp xỉ bao nhiêu tại K đó?",
    "K = 5; LED1 khoảng 11.3",
    all_of=[g(76, "tham số K = 5 là lựa chọn"), g(82, "Ở K = 5: LED1 khoảng 11.3")])
add("test-19", "test", "multihop",
    "Với TimesNet + InceptionBlock V3, số tham số ở độ dài dự báo 100 là bao nhiêu, và MAE trung bình của LED1 là bao nhiêu?",
    "13015 tham số; MAE trung bình LED1 là 158.14",
    all_of=[g(83, "100 13015 0.0507"), g(85, "MAE 128.25 186.51 158.14")])
add("test-20", "test", "multihop",
    "Sản phẩm LED1 là loại sản phẩm nào, và có bao nhiêu điểm bất thường trong tổng số điểm dữ liệu của nó?",
    "Bóng LED Bulb trụ; 72 điểm bất thường trên 1078 điểm",
    all_of=[g(65, "LED1 Bóng LED Bulb TR100N1/30W E27 6500K SS LED Bulb Trụ"),
            g(67, "Xác định 72 điểm bất thường của bộ dữ liệu trong tổng số 1078 điểm")])
add("test-21", "test", "text", "Trong tổng số 1078 điểm dữ liệu của LED1, có bao nhiêu điểm được xác định là bất thường?",
    "72", [g(67, "Xác định 72 điểm bất thường của bộ dữ liệu trong tổng số 1078 điểm")])
add("test-22", "test", "text", "Learning rate khởi tạo và hàm mất mát được dùng trong huấn luyện là gì?",
    "Learning rate 10^-4; hàm mất mát MSE", [g(76, "learning rate khởi tạo là 10−4 và hàm mất mát MSE")])
add("test-23", "test", "text", "Dữ liệu được sắp xếp theo tiêu chí nào trước khi chia thành tập huấn luyện và kiểm tra?",
    "Theo ngày tháng năm, không ngẫu nhiên",
    [g(65, "được sắp xếp theo ngày tháng năm lần lượt chứ không hề ngẫu nhiên")])
add("test-24", "test", "text", "Ngoài độ chính xác, tiêu chí nào được dùng để đánh giá các phiên bản Inception Block?",
    "Độ phức tạp tính toán", [g(82, "Độ phức tạp tính toán là một trong những tiêu chí đánh giá cần thiết")])
absent("test-25", "test", "Nhà máy sản xuất các bóng đèn LED nằm ở đâu?",
       "Tài liệu không nêu địa điểm nhà máy.", ["nhà máy"])
absent("test-26", "test", "Mô hình được cài đặt bằng TensorFlow hay Keras?",
       "Tài liệu không nêu TensorFlow hoặc Keras.", ["TensorFlow", "Keras"])
absent("test-27", "test", "Sản lượng sản xuất của LED5 trong năm 2024 là bao nhiêu?",
       "Tài liệu không có sản lượng sản xuất của LED5.", ["sản lượng sản xuất"])
absent("test-28", "test", "Mô hình được huấn luyện trên đám mây (cloud) nào?",
       "Tài liệu không nêu nền tảng đám mây.", ["đám mây", "cloud"])
absent("test-29", "test", "Mô hình có dự báo bằng dữ liệu thời tiết của khu vực không?",
       "Tài liệu không dùng dữ liệu thời tiết trong thực nghiệm.", ["dữ liệu thời tiết"])
absent("test-30", "test", "Chi phí bản quyền phần mềm để huấn luyện mô hình là bao nhiêu?",
       "Tài liệu không nêu chi phí bản quyền phần mềm.", ["bản quyền"])


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(Q, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {len(Q)} questions to {OUT}")
