# Chạy CarVision AI trên máy mới

Dành cho Windows, Python 3.12, VS Code. Website vẫn chạy bằng Python/Streamlit; `npm start` là lệnh khởi động ngắn.
Code nằm trên Git. Model/cache và ảnh đối chiếu được chia sẻ riêng trong nhóm, không đưa vào commit.

## 1. Nhận đúng phiên bản code

Sau khi Nam công bố nhánh hoặc merge PR, clone vào **thư mục mới**:

```powershell
git clone https://github.com/PoNamVN/AI-Project.git
Set-Location AI-Project
code .
```

Nếu Nam mới gửi nhánh PR, dùng `git switch --track origin/ten-nhanh-Nam-gui`.
Không mặc định `main` đã chứa tuần 4–5 trước khi PR thực sự được merge.

## 2. Cài môi trường nhận diện riêng

Trong terminal VS Code tại thư mục repo:

```powershell
python -m venv .venv-vision
.\.venv-vision\Scripts\python.exe -m pip install --upgrade pip
.\.venv-vision\Scripts\python.exe -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv-vision\Scripts\python.exe -m pip install -r requirements-demo.txt -r requirements-vision.txt
```

Chọn `.venv-vision\Scripts\python.exe` trong **Python: Select Interpreter**.
Không cần Activate hoặc đổi Execution Policy. Không cài TensorFlow vào môi trường này;
nếu làm CNN/MobileNetV2, xem [kiểm thử riêng](../tests/README.md).
Không dùng `pip install -r requirements.txt` cho bước này: đó là danh sách dự kiến cũ, chưa khóa tương thích chung.

## 3. Nhận gói runtime từ Nam

Nhận **hai file cùng phiên bản** qua nơi chia sẻ riêng của nhóm:

- `carvision-runtime-v31.zip`: model/classifier, cache encoder/detector, ảnh train dùng đối chiếu và manifest lịch sử.
- `carvision-runtime-v31.zip.sha256`: checksum ZIP do Nam cung cấp.

Gói khá lớn vì chứa trọng số pretrained. Chỉ máy chủ chạy app hoặc máy thành viên muốn chạy local mới cần nhận gói.
Người dùng mở website không tải gói này. Gói không chứa môi trường Python; bước 2 vẫn cần thiết.
Không có link tải tự động trong repo; chưa cấu hình Supabase cũng chạy được chế độ ảnh tham chiếu local.

Đặt hai file cạnh nhau ngoài repo, sửa đường dẫn dưới đây thành nơi vừa tải:

```powershell
$runtimeZip = 'D:\Downloads\carvision-runtime-v31.zip'
$runtimeExpected = ((Get-Content -LiteralPath ($runtimeZip + '.sha256')) -split '\s+')[0]
$runtimeActual = (Get-FileHash -LiteralPath $runtimeZip -Algorithm SHA256).Hash
if ($runtimeExpected -notmatch '^[a-fA-F0-9]{64}$' -or $runtimeActual -ne $runtimeExpected) {
    throw 'Checksum khong khop. Dung lai va tai dung goi tu Nam.'
}
Expand-Archive -LiteralPath $runtimeZip -DestinationPath .
```

Chỉ giải nén vào bản clone mới, không dùng `-Force` để đè model/dữ liệu đang có.
Nếu máy đã có runtime khác, dùng một bản clone mới để thử, giữ nguyên bản cũ.
Sau giải nén phải có:

```text
data/processed/vision-recognition-v3-brand/model.json
data/processed/vision-recognition-v3-brand/classifier.npz
.kickoff-build/huggingface/...
.kickoff-build/torch/checkpoints/...
data/raw/...                         đúng các ảnh đối chiếu được ghi trong model
.kickoff-build/runtime-bundle-manifest.json
.kickoff-build/runtime-licenses/...
```

Giữ [nguồn và giấy phép](runtime_notices.md) cùng gói. Ảnh tham chiếu là ảnh train với nhãn ứng viên đã dùng phát triển,
không phải dữ liệu mới đã được nhóm duyệt độc lập. Bundle không chứa bộ ảnh ứng viên tuần 5.
Các `review.csv`/`splits.csv` lịch sử được chép nguyên để công cụ đánh giá chặn ảnh đã từng dùng;
không có ảnh của các bộ kiểm tra đi kèm. Gói runtime không thay thế toàn bộ dữ liệu để huấn luyện lại.

## 4. Kiểm tra và chạy

```powershell
npm run check
npm start
```

Không cần `npm install`. Thiếu Node/npm thì chạy trực tiếp:

```powershell
.\.venv-vision\Scripts\python.exe -m src.check_environment
.\.venv-vision\Scripts\python.exe -m streamlit run app/app.py
```

Mở Local URL terminal đưa ra, thường `http://localhost:8501`.
`npm run check` kiểm tra tệp và thư viện, không đo độ chính xác và không bảo đảm mọi weights load được.
Chọn ảnh một xe đã biết để kiểm tra thật: xem Top-3, ba ảnh đối chiếu, lưu lựa chọn và tải JSON.
Nếu camera hoặc Supabase chưa cấu hình, vẫn có thể chọn ảnh từ máy và dùng nguồn **Trên máy chủ**.

Để kiểm tra CLI bằng một ảnh train có sẵn trong gói, không tạo bộ test hay accuracy mới:

```powershell
$runtimeModel = Get-Content data/processed/vision-recognition-v3-brand/model.json -Raw | ConvertFrom-Json
$runtimePhoto = $runtimeModel.train_records | Where-Object class_id -eq 'model_s' | Select-Object -First 1
.\.venv-vision\Scripts\python.exe -m src.recognize_vehicle data/processed/vision-recognition-v3-brand --image (Join-Path 'data/raw' $runtimePhoto.source_path) --output data/processed/team-smoke-v1
```

Mỗi lần chạy cần output mới. Không lấy việc chạy thành công trên ảnh train làm bằng chứng accuracy ngoài thực tế.

## 5. Khi gặp lỗi

| Thông báo | Cách xử lý |
|---|---|
| Chưa có môi trường Python | Làm bước 2 tại đúng thư mục repo |
| Thiếu model/cache | Kiểm tra checksum và vị trí giải nén, không huấn luyện lại chỉ để mở app |
| Ảnh tham chiếu thiếu/khác checksum | Lấy đúng gói runtime cùng model; không sửa checksum để bỏ qua |
| Supabase chưa cấu hình | Chọn **Trên máy chủ**; chỉ owner nhập khóa theo tài liệu Supabase |
| Cổng 8501 bận | `npm start -- --server.port 8502` |
| Có nhiều xe trong ảnh | Chọn đúng số khung xe trong giao diện |

Không gửi khóa, `.env`, ảnh riêng hoặc toàn bộ model trong issue công khai.

## 6. Nam tạo lại gói khi model thay đổi

Chạy trên máy đang có đủ runtime và ảnh gốc tương ứng, chọn **tên ZIP mới** mỗi lần:

```powershell
python scripts/build_runtime_bundle.py --output .kickoff-build/handoff/carvision-runtime-v31.zip
```

Script chỉ lấy tệp runtime được chỉ định và ảnh trong `train_records`, kiểm tra checksum ảnh, giữ nguồn/giấy phép,
không xuất secret hoặc toàn bộ thư mục dự án, không huấn luyện và không ghi đè bundle đã tồn tại.
Gửi ZIP và checksum qua nơi chia sẻ riêng của nhóm. Code vẫn đi qua nhánh riêng/PR theo CONTRIBUTING.
