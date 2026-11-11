# Tuần 3 · Rà ảnh và chuẩn bị dữ liệu

## Lấy bản đã chạy của nhóm

Sau khi cập nhật code bằng `git pull --ff-only origin main`, tải [week3-baseline-artifacts-v1.zip](https://github.com/PoNamVN/AI-Project/releases/download/dataset-week3-v1/week3-baseline-artifacts-v1.zip). Giải nén vào thư mục gốc repo để có bảng rà, bộ ảnh chuẩn hóa và kết quả trong `data/processed/`. Nếu các đường dẫn đó đã tồn tại trên máy, giải nén vào thư mục khác trước để đối chiếu và giữ bản cũ.

ZIP có bảng rà 91 ảnh, 66 ảnh đã chuẩn hóa, `splits.csv`, `summary.csv`, `baseline.keras`, `metrics.json`, `test_predictions.csv`, thông tin ghi công và danh sách thư viện của lần chạy. Có thể mở báo cáo ngay hoặc dùng bộ ảnh chuẩn hóa để chạy lại CNN với tên đầu ra mới theo mục 3. Giữ tệp ghi công cùng ảnh khi chia sẻ.

Chạy lệnh ở thư mục gốc repo trong PowerShell. Dùng môi trường đã cài `requirements-demo.txt` theo README. Các lệnh dưới đây đọc bộ ảnh ứng viên trong `data/raw` và lưu đầu ra ở `data/processed`.

## 1. Tạo bảng rà soát

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset review data/raw/tesla-week3-candidate --output data/processed/week3_review.csv
```

Mở CSV bằng Excel hoặc VS Code. Xem từng ảnh trong Data Studio hoặc bằng đường dẫn `relative_path`. Nếu Excel mở file với sai dấu phân cách, nhập dưới dạng CSV UTF-8 phân cách bằng dấu phẩy. Khi lưu lại, giữ định dạng CSV.

- Đặt `decision` thành `keep` khi ảnh đúng dòng, đủ rõ và nguồn/giấy phép đã được rà. Đặt `exclude` cho ảnh loại hoặc chưa đủ căn cứ dùng trong lần chia này. `pending` nghĩa là chưa ra quyết định.
- `class_id` ban đầu lấy từ thư mục ảnh. Chỉ sửa sau khi xác nhận dòng xe. Dùng một trong các mã `model_s`, `model_3`, `model_x`, `model_y`, `cybertruck`.
- Điền `vehicle_id` cho mỗi ảnh giữ lại, ví dụ `x-001`. Các góc của cùng một xe hoặc các ảnh có khả năng là cùng xe phải dùng chung mã. Chỉ cấp mã riêng khi đã rà các ảnh có thể liên quan. Chọn một ảnh đại diện nếu ảnh gần trùng và loại các ảnh trùng chính xác dư thừa.
- Giữ nguyên `relative_path` và `sha256`. Nếu bộ ảnh thay đổi, tạo bảng mới với tên khác và rà lại các quyết định bị ảnh hưởng.

## 2. Chuẩn hóa và chia tập

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset split data/raw/tesla-week3-candidate --review data/processed/week3_review.csv --output data/processed/tesla-week3-v1 --seed 42
```

Lệnh yêu cầu mọi ảnh đã có quyết định. Giữ ít nhất hai dòng xe và ít nhất ba nhóm xe cho mỗi dòng giữ lại. Đây là điều kiện tối thiểu để có đủ ba tập, chưa phải cỡ mẫu đủ để kết luận độ chính xác. Nếu giảm số dòng, ghi phạm vi đó trong báo cáo.

Chia theo nhóm xe trong từng dòng, gần 70/15/15 tính theo số nhóm. Tỷ lệ theo số ảnh có thể khác. Cùng bảng rà và cùng seed tạo cùng cách chia. Ảnh đầu ra là PNG RGB 128 × 128, sửa hướng theo EXIF và thêm viền khi cần để giữ tỷ lệ xe.

- `train/`, `validation/`, `test/` chứa ảnh đã chuẩn hóa theo dòng.
- `splits.csv` lưu đường dẫn ảnh chuẩn hóa, ảnh nguồn, mã xe, hash ảnh gốc, nguồn và giấy phép.
- `summary.csv` đếm ảnh theo dòng và theo tập.

Giữ bảng rà, seed và danh sách chia cùng phiên bản dữ liệu khi chạy CNN baseline. Dùng validation để chọn cấu hình. Chỉ đo test sau khi chốt cấu hình. Bộ ảnh gốc vẫn giữ riêng cùng `provenance.csv` đầy đủ để tra thông tin ghi công. Đặt tên đầu ra mới nếu chạy phiên bản khác.

## 3. Huấn luyện CNN baseline

Dùng môi trường Python 3.12 riêng cho TensorFlow CPU. Trên Windows, mở terminal ở thư mục gốc và chạy:

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv-ml\Scripts\python.exe -m src.train_baseline data/processed/tesla-week3-v1 --output data/processed/week3-baseline-v1 --seed 42 --epochs 25
```

CNN có ba tầng tích chập, tự học dấu hiệu từ ảnh và dùng Adam với tốc độ học 0,001. Mỗi lần cập nhật dùng tối đa 8 ảnh. Chương trình giữ trọng số tại epoch có loss validation thấp nhất và dừng sớm sau 5 epoch không cải thiện. Sau khi chốt trọng số mới đọc ảnh test để đánh giá.

- `baseline.keras` chứa mô hình đã huấn luyện.
- `metrics.json` chứa cấu hình, mã kiểm tra của danh sách chia, lịch sử học, accuracy, macro-F1 và ma trận nhầm lẫn. Ma trận có hàng là nhãn thật, cột là nhãn dự đoán, cùng thứ tự `class_ids`.
- `test_predictions.csv` ghi kết quả cho từng ảnh. Đối chiếu đường dẫn với `splits.csv` để tìm ảnh nguồn.

Báo cáo cũng có mốc so sánh luôn đoán dòng có nhiều ảnh nhất trong train. Dữ liệu nhỏ chỉ đủ cho thử nghiệm ban đầu. Giữ nguyên tập test khi so sánh mô hình tuần 4, và dùng validation để chọn cấu hình. Muốn tái lập lần chạy hiện tại, cần cả ZIP nguồn và cùng bảng rà soát.

### Kết quả chạy thử ngày 04/11/2026

Từ 91 ảnh ứng viên, bảng rà đề xuất giữ 66 và loại 25. Chia theo nhóm xe với seed 42, thu được 45 ảnh train, 7 ảnh validation và 14 ảnh test. CNN chạy 15 epoch và giữ trọng số epoch 10 theo loss validation. Trên test, CNN đoán đúng 7/14 ảnh (50,0%), macro-F1 là 25,6%. Mô hình đoán đúng 2 ảnh Model 3 và 5 ảnh Model Y. Cả 7 ảnh thuộc ba lớp còn lại đều bị đoán nhầm.

Model 3 và Model Y cùng có 12 ảnh train. Mốc luôn đoán lớp nhiều nhất chọn Model 3 theo thứ tự mã lớp và đúng 2/14 ảnh. Nếu chọn Model Y trong trường hợp đồng hạng, mốc đó đúng 5/14 ảnh. Chỉ số có thể thay đổi mạnh trên mẫu nhỏ; mã nhóm xe là ước tính từ ảnh và tên nguồn.
