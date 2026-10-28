# Week 2 — Tesla Data Studio

## Đã triển khai

- Giao diện Streamlit tiếng Việt: biểu đồ theo lớp, bảng chất lượng, xem ảnh và tải báo cáo.
- Đọc dữ liệu theo thư mục lớp hoặc `train/val/test` có thư mục lớp bên trong.
- Kiểm tra ảnh giải mã lỗi, kích thước, nhãn thiếu/lạ và nguồn chưa khai báo.
- Tìm ảnh trùng byte và pixel sau giải mã/áp dụng EXIF; cảnh báo nhãn khác nhau trên ảnh trùng.
- Cảnh báo nhóm ảnh trùng hoặc cùng `vehicle_id` xuất hiện ở nhiều tập.
- Xuất CSV an toàn hơn khi mở bằng bảng tính và JSON đầy đủ. Không xóa/sửa/chia ảnh tự động.

## Chuẩn bị dữ liệu

Chỉ đưa ảnh được phép dùng vào `data/raw/tesla/`:

Ảnh trên 25 triệu pixel được đánh dấu không đọc để giới hạn bộ nhớ; rà soát hoặc chuẩn bị bản kích thước phù hợp riêng, công cụ không sửa bản gốc.

```text
data/raw/tesla/
  model_s/
  model_3/
  model_x/
  model_y/
  cybertruck/
  provenance.csv    (tùy chọn)
```

Chấp nhận tên `Model 3`, `model_3`, `model-3`, `Model3` và các tên tương ứng. Ảnh nằm trực tiếp ở gốc bị đánh dấu thiếu nhãn; thư mục khác năm lớp bị đánh dấu nhãn lạ. Định dạng hỗ trợ: JPG/JPEG, PNG, WEBP, BMP, TIFF. File khác không được tính là ảnh.

Nếu đã chia tập, dùng `train/model_3/...`, `val/model_3/...`, `test/model_3/...` ngay dưới gốc đã chọn. Không trộn các cách đặt thư mục nếu không cần. Công cụ không tự chứng nhận việc chia tập đúng chỉ vì không thấy ảnh trùng.

### Nguồn và mã xe

Tạo `provenance.csv` trong gốc dataset với các cột:

```csv
relative_path,source_url,license,vehicle_id
```

Mỗi dòng là một ảnh, đường dẫn tương đối từ gốc dataset, dùng dấu `/`. `source_url` là nguồn gốc của ảnh, `license` ghi điều kiện sử dụng cần kiểm chứng. `vehicle_id` là mã một chiếc xe thực tế, không phải mã dòng xe; không biết thì để trống. CSV thiếu cột bắt buộc, đường dẫn ngoài dataset hoặc bản ghi trùng sẽ báo lỗi rõ ràng. Không có manifest vẫn quét được, nhưng ảnh thiếu nguồn sẽ được gắn cờ.

## Chạy bằng VS Code

Đứng ở gốc repo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Ở sidebar nhập `tesla` nếu ảnh ở `data/raw/tesla/`, rồi bấm **Quét dữ liệu**. Ứng dụng chỉ cho chọn thư mục trong `data/raw`; không nhập URL hoặc đường dẫn ảnh riêng lẻ. Sau khi đổi ảnh/nhãn, bấm quét lại để cập nhật snapshot. Báo cáo được tải về qua trình duyệt khi bạn bấm nút; app không âm thầm ghi ra thư mục khác.

### Dòng lệnh

```powershell
.\.venv\Scripts\python.exe -m src.data_audit data/raw/tesla --output reports/audit
```

Mỗi lần chạy tạo một thư mục báo cáo mới theo thời gian UTC với `audit.json` và `images.csv`. Thư mục báo cáo phải nằm ngoài thư mục ảnh để tránh lẫn dữ liệu. Báo cáo cũ không bị ghi đè.

## Đọc kết quả đúng

- Số ảnh là số file có đuôi hỗ trợ thực sự tìm thấy trong lần quét.
- Số ảnh theo lớp gồm ảnh đọc được, kể cả ảnh trùng; không phải số ảnh độc lập đã duyệt.
- Một ảnh có thể có nhiều vấn đề; không cộng các mục thành tổng ảnh lỗi.
- Ảnh hỏng vẫn được ghi trong CSV; không lấy đó làm ảnh huấn luyện.
- Nhãn thư mục và giấy phép do người chuẩn bị khai báo; công cụ không chứng minh chúng đúng.
- Nhóm trùng byte và pixel có thể mô tả cùng các file; không cộng số nhóm để suy ra số xe.
- Chưa có ảnh thì không có biểu đồ/kết quả giả. Thống kê bằng 0 sau quét thư mục trống không phải audit một dataset hoàn chỉnh.

## Chưa làm và giới hạn

Chưa tải dataset, kiểm duyệt quyền ảnh, xác minh tên xe bằng AI, tìm mọi ảnh gần giống, chia tập tự động, huấn luyện CNN/MobileNetV2 hay tích hợp thông số. Hash pixel chỉ tìm ảnh giải mã giống hệt; resize, crop hoặc ảnh chụp cùng xe ở góc khác vẫn cần kiểm tra bằng mắt hoặc `vehicle_id` đúng. Độ đa dạng ánh sáng, góc nhìn và bias cần nhóm phân tích thêm từ ảnh thật.

Không dùng kết quả kiểm thử tổng hợp như số liệu dataset của nhóm. Công cụ cần được chạy trên dataset thực sau khi xác minh nguồn để có bằng chứng Week 2.

## Kiểm thử code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Test sinh ảnh nhỏ trong thư mục tạm, kiểm tra nhãn, file lỗi, trùng, rò rỉ giữa tập, manifest, xuất CSV và luồng giao diện. Không cần tải dataset thật để chạy test.
