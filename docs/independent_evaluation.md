# Đánh giá bằng ảnh mới

Mẫu manifest trống: [field_evaluation.csv](field_evaluation.csv). Không có ảnh hay kết quả giả.
Đây là quy trình kiểm tra một ảnh đầu vào qua toàn luồng V3.1; chưa đánh giá lợi ích của hai góc ảnh.

1. Thu ảnh mới có quyền sử dụng, không lấy lại các batch từng dùng phát triển. Chụp nhiều xe vật lý,
   góc nhìn, ánh sáng, khoảng cách và các hãng ngoài Tesla; rà nhãn trước khi chạy model.
2. Tạo thư mục mới `data/raw/field-batch-v1/`, giữ ảnh trong các thư mục `model_s`, `model_3`,
   `model_x`, `model_y`, `cybertruck`, hoặc `other_car`. Không sửa hoặc ghi đè bộ cũ.
3. Sao chép mẫu CSV vào một nơi riêng, điền mỗi ảnh một dòng. `source_path` có dạng
   `field-batch-v1/model_3/ten-anh.jpg`, tương đối với `data/raw`. `vehicle_id` gom các góc của cùng xe;
   không tự coi nhiều ảnh cùng xe là nhiều xe độc lập. `source_url` có thể là mã ghi nhận tự chụp;
   `license` ghi quyền thực tế. Nhãn/mã xe do người rà chịu trách nhiệm, chưa tự được xác minh bởi công cụ.
4. Chốt manifest và cấu hình trước khi dự đoán. Không dùng điểm của batch này để chọn model rồi tiếp tục
   báo nó là test mới. Công cụ lưu bản manifest và checksum model trong output.

```powershell
.\.kickoff-build\vision-v3\Scripts\python.exe -m src.evaluate_field data/processed/field-batch-v1.csv --output data/processed/field-evaluation-v1 --labels-reviewed
```

Cần môi trường/model/cache và các manifest cũ được giữ: `tesla-expanded-v2/splits.csv`,
`week4-newcheck-v1/review.csv`, `non-tesla-gate-review-v1/review.csv`, `non-tesla-brand-training-v3/review.csv`.
Lệnh chỉ đánh giá, không học lại, không đổi ngưỡng và không ghi đè output đã có.

## Kiểm tra dữ liệu trước, không nạp AI

Thêm `--validate-only` để kiểm tra ảnh đọc được, kích thước/pixel, đường dẫn, nguồn/quyền ảnh,
mã xe và checksum. Chế độ này chỉ cần metadata model cùng thư viện đọc ảnh, không cần weights hoặc cache encoder:

```powershell
.\.kickoff-build\vision-v3\Scripts\python.exe -m src.evaluate_field data/processed/field-batch-v1.csv --output data/processed/field-validation-v1 --labels-reviewed --validate-only
```

Đầu ra là `validation.json` và bản manifest, **không có accuracy hoặc dự đoán**.
Chỉ dùng `--labels-reviewed` sau khi người phụ trách đã rà nhãn, nguồn và quyền dùng ảnh.
Kiểm tra trùng đọc checksum của tất cả `review.csv`/`splits.csv` trong các thư mục trực tiếp dưới `data/processed`,
kể cả ảnh xe khác dùng học gate; không lấy manifest đang kiểm tra làm dữ liệu lịch sử của chính nó.

`metrics.json` báo số ảnh/mã xe khai báo, phân bố nhãn/trạng thái, Tesla Top-1 đúng trên **mọi ảnh Tesla đầu vào**,
Top-3 đúng và số ảnh ngoài Tesla còn bị trả gợi ý Tesla. Ảnh không phát hiện được xe hoặc cần chọn xe vẫn nằm
trong mẫu số Tesla. Không gộp tỷ lệ từ chối xe khác với độ chính xác phân loại dòng xe để làm đẹp kết quả.

Kiểm tra SHA-256 loại ảnh trùng trong batch và ảnh trùng nội dung với các manifest phát triển được kiểm tra.
Ảnh crop/đổi kích thước hoặc cùng xe chụp góc khác có thể vượt kiểm tra hash; cần rà thủ công trước.
Không thể chứng minh ảnh chưa từng xuất hiện trong dữ liệu pretrained. Mỗi lớp ít ảnh chưa đủ kết luận tổng quát.

Hiện chưa có batch đường phố mới được cung cấp/duyệt nên **chưa có accuracy mới**.
