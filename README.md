# Ứng dụng dự đoán nguy cơ thiếu hàng (Backorder) – Đề tài 10

Ứng dụng Streamlit dùng mô hình **Random Forest – Luồng A (Feature Engineering)** đã huấn luyện từ bộ dữ liệu `clu.zip`.

## 1. Tính năng

- Dự đoán nguy cơ thiếu hàng cho một sản phẩm qua biểu mẫu.
- Tải CSV để dự đoán hàng loạt và tải kết quả về CSV. Kết quả lưu xác suất đầy đủ, ngưỡng chính xác và chiến lược ngưỡng đã dùng.
- Chọn ngưỡng cảnh báo: F1 tối đa, ưu tiên Recall >= 80%, hoặc ngưỡng mặc định 0,50.
- Chuẩn hóa các cột cờ nhận giá trị `Yes/No`, `Y/N`, `True/False` hoặc `0/1`.
- Tự động bỏ dòng tổng kết cuối CSV dạng `(242075 rows)` nếu có.
- Hiển thị gợi ý kiểm tra tồn kho, hàng đang vận chuyển và hàng quá hạn.

## 2. Chạy trên WSL (Ubuntu)

Giải nén ZIP trong WSL, sau đó mở Terminal tại thư mục `De_tai_10_Web`:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Mở `http://localhost:8501` trong trình duyệt Windows. Giữ Terminal WSL mở trong khi sử dụng ứng dụng; nhấn `Ctrl+C` để dừng.

## 3. Chạy trên Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 4. Dùng CSV

CSV cần có đủ 21 cột thuộc tính đầu vào. Cột `sku` là tùy chọn. Cột `went_on_backorder` nếu có sẽ không được dùng làm đầu vào dự đoán. Cột cờ có thể ở dạng `Yes/No` hoặc `0/1`.

## 5. Tiền xử lý

- Mã `-99` trong `perf_6_month_avg` và `perf_12_month_avg` được xem như dữ liệu thiếu, giống quy trình huấn luyện.
- Dùng median imputer và RobustScaler đã fit trên tập train; không fit lại trên dữ liệu tải lên.
- Sáu đặc trưng bổ sung được tạo theo Luồng A.
- Mô hình dự đoán nguy cơ backorder, chưa tính lượng đặt hàng tối ưu, chi phí tồn kho hay lịch vận chuyển.

## 6. Các ngưỡng khảo sát trên Validation

| Chiến lược | Ngưỡng | Precision | Recall | F1-score | FN |
|---|---:|---:|---:|---:|---:|
| Mặc định | 0,500 | 0,696 | 0,693 | 0,695 | 139 |
| F1 tối đa | 0,418 | 0,655 | 0,764 | 0,705 | 107 |
| Ưu tiên Recall >= 80% | 0,361 | 0,613 | 0,801 | 0,695 | 90 |

Các ngưỡng được lựa chọn từ Validation của bộ dữ liệu dự án; cần đánh giá lại trên dữ liệu độc lập trước khi triển khai thực tế. Giao diện hiển thị ngưỡng với 9 chữ số thập phân. CSV kết quả có thêm `nguong_ap_dung` và `chien_luoc_nguong`; xác suất không bị làm tròn trước khi xuất file, vì vậy nhãn `du_doan` có thể đối chiếu chính xác với ngưỡng.

## 7. Tính nhất quán của ngưỡng và file kết quả

- Ngưỡng được hiển thị với 9 chữ số thập phân để tránh hiểu nhầm do làm tròn.
- File CSV xuất ra có thêm hai cột `nguong_ap_dung` và `chien_luoc_nguong`.
- `xac_suat_thieu_hang` được lưu bằng độ chính xác số thực gốc; không làm tròn trước khi tạo nhãn.
- Có thể kiểm tra nhãn bằng điều kiện `du_doan = 1` khi và chỉ khi `xac_suat_thieu_hang >= nguong_ap_dung`.
