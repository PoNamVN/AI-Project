# Bản chạy CPU gọn — 25/11/2026

CarVision vẫn nhận diện năm dòng Tesla, dùng classifier V3.1 và checkpoint DINOv2/SigLIP2 cũ.
Không thêm hãng mới, không học lại, không dùng 29 ảnh ứng viên chưa được nhóm duyệt.

## Bản mới thay đổi gì?

- ONNX Runtime CPU thay cho PyTorch/Transformers khi suy luận; giao diện vẫn là Streamlit.
- Nén **chỉ trọng số MatMul** theo block 32, 8 bit, `accuracy_level=1`; giữ detector dạng float.
- Tiền xử lý bằng Pillow/NumPy; gần tương đương Torchvision uint8, không tuyên bố giống từng số.
- Tái sử dụng logic chọn xe, từ chối ngoài phạm vi, xếp hạng, đối chiếu và giải thích hiện có.
- Ảnh có khung xe minh họa được thu nhỏ trước khi giữ trong phiên; tọa độ kết quả vẫn theo ảnh gốc.
- Runtime kiểm checksum từng tệp, checkpoint nguồn và liên kết với classifier/metadata.
  Bản mặc định chỉ nạp khi có kết quả validation khớp đúng manifest runtime.

## Kết quả đã đo

So sánh trực tiếp cùng **29 ảnh validation Tesla + 9 ảnh validation ngoài Tesla** đã dùng phát triển:

| Chỉ số toàn luồng | PyTorch image-only | ONNX chỉ nén trọng số |
|---|---:|---:|
| Tesla Top-1 đúng trên toàn bộ 29 đầu vào | 25/29 | 25/29 |
| Macro-F1 Tesla trên toàn bộ đầu vào | 0,8827 | 0,8827 |
| Tesla Top-3 đúng | 27/29 | 28/29 |
| Xe ngoài Tesla còn nhận gợi ý Tesla | 1/9 | 1/9 |

Top-1 khớp trên cả 38 ảnh. Một vài trạng thái không chắc chắn gần ngưỡng có thay đổi;
điểm vẫn không phải xác suất đã hiệu chuẩn. **25/29 ở đây là toàn luồng**, có tính cả trường hợp
nhiều xe chưa chọn đối tượng; không trộn với số 26/29 classifier-only trong báo cáo cũ.

Bản dynamic INT8 đầu tiên chỉ đúng 17/29, đã bị loại. Bản float ONNX đạt 25/29 nhưng nặng hơn.
Hai thử nghiệm thêm về cách tính/cache không tạo lợi ích phù hợp để chọn thay bản trên.
Không đổi ngưỡng theo test và không gọi những kết quả này là accuracy ngoài thực tế.

Phép thử giao diện thật trên Windows, dùng một ảnh train cũ và ảnh Supabase thật:

| Chỉ số | Trước, PyTorch image-only | Bản ONNX |
|---|---:|---:|
| Peak resident RAM | 1.187 MiB | 401 MiB |
| Lượt đầu gồm nạp model và ảnh đối chiếu | 19,51 giây | 10,28 giây |
| Phiên thứ hai dùng lại model/cache | 1,34 giây | 1,41 giây |

Đây là số đo local theo các lượt riêng, không phải benchmark tải hoặc cam kết độ trễ.
Bản ONNX có private memory khoảng 687 MiB trong phép thử trên Windows;
chưa đo cgroup Linux, ảnh lớn đồng thời, camera điện thoại qua HTTPS hoặc Render Free.
**Không kết luận chắc chạy được trong 512 MB chỉ dựa vào resident RAM 401 MiB.**
Luồng chọn mẫu, không phù hợp, JSON và cache ảnh hoạt động; không import Torch trong phép thử ONNX.

Đã thử cài ZIP vào một thư mục sạch chỉ chứa mã nguồn và runtime: không có `.env`, cache Torch
hoặc ảnh dataset. Máy chủ khởi động đạt health check; nhận diện một ảnh train cũ, giải thích bằng
che 9 vùng và lấy 3 thumbnail Supabase đều thành công. Cấu hình Supabase chỉ truyền qua môi trường
tiến trình thử, không sao chép vào gói. Toàn bộ 58 kiểm thử: 54 đạt, 4 bỏ qua do không cài TensorFlow;
3 kiểm thử Node đều đạt. Quy trình export rồi validation cũng đã chạy lại thành công từ mã nguồn.

## Chạy trên máy Nam

Sau khi dừng phiên cũ bằng Ctrl+C:

```powershell
npm start
```

Launcher tự chọn `.venv-onnx` khi gói `data/processed/vision-onnx-v31` đã qua validation.
Muốn chọn rõ: `npm run start:lite`; trở lại bản PyTorch: `npm run start:torch`.
Thêm cổng như trước: `npm start -- --server.port 8502`.
Không cần gửi khóa mới hoặc tải toàn kho ảnh. `.env` riêng được launcher đọc như trước.

## Máy thành viên / máy chủ mới

```powershell
python -m venv .venv-onnx
.\.venv-onnx\Scripts\python.exe -m pip install -r requirements-onnx.txt
```

Nhận ZIP compact cùng SHA-256 riêng từ Nam. Gói hiện tại:
`carvision-compact-runtime-v31.zip`, 150.913.005 bytes (~151 MB), 11 tệp runtime/giấy phép và một manifest của gói.
Không có ảnh dataset, `.env`, cache PyTorch hoặc môi trường Python.
SHA-256:

```text
051e1b7d37dc2b06f088bbd5e05caf6cbd50a77f78cca9fef929f29e6544dc08
```

Cài gói bằng công cụ kiểm checksum/đường dẫn (thay đường dẫn ZIP thực tế):

```powershell
.\.venv-onnx\Scripts\python.exe -m src.install_runtime --archive D:\Downloads\carvision-compact-runtime-v31.zip --sha256 051e1b7d37dc2b06f088bbd5e05caf6cbd50a77f78cca9fef929f29e6544dc08
npm run setup:cloud
npm run check
npm start
```

Runtime compact dùng ảnh đối chiếu Supabase; máy chủ cần cấu hình riêng. Không đưa secret vào chat/Git.
Gói chưa upload nơi chia sẻ hoặc GitHub; thay đổi code cũng chưa commit/push.

## Tái tạo dành cho người phát triển

Giữ môi trường export riêng; môi trường chạy web không cài các thư viện export/huấn luyện:

```powershell
python -m venv .kickoff-build\onnx-export-env
.\.kickoff-build\onnx-export-env\Scripts\python.exe -m pip install -r requirements-onnx-export.txt
.\.kickoff-build\onnx-export-env\Scripts\python.exe scripts/export_compact_runtime.py --output data/processed/onnx-trial-v1
.\.kickoff-build\onnx-export-env\Scripts\python.exe scripts/validate_compact_runtime.py data/processed/onnx-trial-v1 --output data/processed/onnx-trial-validation-v1
```

Cần cache checkpoint nguồn, classifier và manifest/ảnh validation cũ trên máy export.
Export không tự tạo dấu đã đạt validation. Chỉ đóng gói khi gate giữ các ca Top-1 đúng,
không giảm macro-F1/Top-3 và không tăng gán nhầm Tesla trên cùng validation.
Mọi output phải mới; không ghi đè ảnh, runtime hoặc kết quả cũ.

```powershell
.\.venv-onnx\Scripts\python.exe scripts/build_runtime_bundle.py --compact --output .kickoff-build/compact-new-v1/carvision-compact-runtime-v31.zip
```

Lệnh đóng gói dùng runtime đã chốt ở `data/processed/vision-onnx-v31`; không tự đổi bản đang dùng theo trial.
Giữ ảnh kiểm tra độc lập cho đánh giá sau; dữ liệu ít hoặc lệch nhãn vẫn là giới hạn chính về khả năng tổng quát.
