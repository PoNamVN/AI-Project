# Đưa CarVision AI lên Render

> Quyết định hiện tại của Nam: bài tập trên trường, chỉ dùng phương án miễn phí.
> Chưa tạo dịch vụ Render có phí. Các cấu hình trả phí phía dưới chỉ giữ để tham khảo;
> không áp dụng mẫu Blueprint trong phạm vi ngân sách hiện tại.

Đã có [runtime CPU gọn hơn](compact_runtime.md), peak resident trong phép thử giao diện Windows khoảng
401 MiB. Chưa kiểm tra Linux/Render Free và private memory vẫn cao hơn resident;
không suy từ số này rằng 512 MB đã đủ. Các số PyTorch/gói 1,64 GB bên dưới là bản đối chứng cũ.

Trạng thái 25/11/2026: đã chuẩn bị code, cấu hình mẫu và gói runtime; **chưa tạo dịch vụ có phí,
chưa có URL công khai, chưa kiểm thử Linux/Render hoặc camera điện thoại qua HTTPS**.
Các file mới vẫn ở nhánh làm việc, chưa có trên `main`.

## 1. Cấu hình và chi phí cần quyết định

Render hỗ trợ Python Web Service, HTTPS và cổng `PORT`. CarVision tiếp tục dùng Streamlit;
Supabase chỉ giữ kho ảnh riêng, máy chủ Render giữ mô hình và xử lý ảnh người dùng.

Sau khi chỉ nạp nhánh ảnh SigLIP cho nhận diện, phép thử trên Windows với một ảnh train cũ:
RAM resident đỉnh khoảng 1,187 MiB (trước đó 2,239 MiB, giảm khoảng 47%),
private memory sau chạy khoảng 1,473 MiB, peak pagefile khoảng 3,746 MiB.
Đối chiếu năm ảnh train cũ: toàn bộ kết quả và đặc trưng ảnh giống bản nạp cả nhánh văn bản,
sai khác đặc trưng tối đa bằng 0. Không thay checkpoint, trọng số hoặc ngưỡng; đây không phải accuracy mới.
Đường huấn luyện/tạo đặc trưng văn bản vẫn dùng engine đầy đủ.
Đây không phải số đo container Linux hoặc thử tải nhiều người. Các lượt inference được xếp lần lượt
trong một process để tránh nhân bộ nhớ trung gian. Không giới hạn hay chứng minh được tổng số phiên người dùng.

Theo tài liệu giá kiểm tra ngày 25/11/2026:

| Gói web | RAM | Giá compute/tháng | Đánh giá cho bản hiện tại |
|---|---:|---:|---|
| Free | 512 MB | $0 | Quá ít so với mức RAM vừa đo |
| `1c-2g` | 2 GB | $25 | Có thể thử sau tối ưu; chưa chứng minh đủ cho nạp model/ảnh lớn/nhiều phiên |
| `2c-4g` | 4 GB | $85 | Mẫu cấu hình chọn để có dư địa; vẫn cần đo trên Render |
| `2c-8g` | 8 GB | $135 | Chỉ cân nhắc nếu số đo thực tế cho thấy cần, chưa có lý do mua ngay |

Đây là phí compute, không phải báo giá tổng gồm mọi sử dụng/thuế. Workspace Hobby có giá nền $0,
không cần tự nâng workspace lên Pro chỉ để dùng web service có phí. Không tạo dịch vụ trước khi Nam chốt ngân sách.
Xem [giá chính thức](https://render.com/pricing.md), [compute plans](https://render.com/docs/compute-plans.md)
và [giới hạn Free](https://render.com/docs/free).

## 2. Gói mô hình riêng cho máy chủ

Tạo gói khi đã có runtime local đầy đủ:

```powershell
.\.kickoff-build\vision-v3\Scripts\python.exe scripts/build_runtime_bundle.py --cloud-only --output .kickoff-build/render-ready-v1/carvision-cloud-runtime-v31.zip
```

Không ghi đè file đã có. Dùng tên thư mục/phiên bản mới khi tạo lại.
Gói cloud chỉ có 15 tệp model/cache/giấy phép, không chứa ảnh dataset, `.env`, nhãn đánh giá mới hoặc code app.
Code app được lấy riêng từ Git. Gói ngày 25/11 có 1,638,394,355 bytes (~1.64 GB), SHA-256:

```text
542e9ce5d3ecb7d95125bfbbee613a57ea972dcebbdc65348848cd3cb6bf0d9c
```

Chủ dự án cần đặt gói vào nơi cho tải qua HTTPS, rồi nhập URL tải vào biến riêng trên Render.
Nếu dùng URL có hạn, đảm bảo còn hiệu lực trong lần build và cấp lại trước khi rebuild khi đã hết hạn.
Không commit URL có quyền truy cập hoặc đưa nó vào chat/log. Kho ảnh Supabase hiện chỉ nhận ảnh tối đa 30 MB;
không thay cấu hình kho ảnh để cố nhét ZIP 1.64 GB vào đó. Chưa upload/công khai gói runtime này.

`src.install_runtime` kiểm SHA-256 toàn ZIP, từng tệp và đường dẫn trước khi cài; không ghi đè file khác nội dung,
không cài ảnh gốc kể cả khi nhận gói runtime nội bộ đầy đủ. Gói tải lỗi phải làm build thất bại.

## 3. Các bước trên Render sau khi code được duyệt/push

1. Tạo **Web Service**, chọn repo `PoNamVN/AI-Project` và đúng nhánh đã có thay đổi này.
2. Chọn Python, Singapore; duyệt lại gói compute và chi phí trước khi tạo.
3. Build command: `pip install -r requirements-render.txt && python -m src.install_runtime`.
4. Start command: `python -m src.serve_public`.
5. Health check path: `/_stcore/health`. Health check HTTP không thay thế thử nhận diện thật.
6. Tắt auto deploy để chọn thời điểm công bố sau khi kiểm tra. Mẫu tương đương nằm ở
   [render.yaml.example](../render.yaml.example); chỉ dùng sau khi đã chốt gói có phí.
7. Nhập các biến sau trong Render Dashboard, không đưa secret vào Git:

| Biến | Giá trị |
|---|---|
| `PYTHON_VERSION` | `3.12.14` |
| `CARVISION_SUPABASE_URL` | URL project CarVision AI |
| `CARVISION_SUPABASE_KEY` | Khóa mới, chỉ nhập riêng trên máy chủ |
| `CARVISION_RUNTIME_URL` | URL HTTPS tải gói runtime |
| `CARVISION_RUNTIME_SHA256` | SHA-256 gói đã chốt |
| `CARVISION_ENABLE_DEV_TOOLS` | `0` |
| `HF_HUB_OFFLINE` | `1` |
| `TOKENIZERS_PARALLELISM` | `false` |

`src.serve_public` bắt buộc cấu hình cloud, kiểm tra runtime và buộc tắt công cụ nội bộ dù biến dev bị đặt nhầm.
Ảnh người dùng không được ghi ra kho ảnh hoặc tự đưa vào huấn luyện. App dùng cache ảnh đối chiếu chung tối đa
32 bản nhỏ trong 15 phút; cache tách theo dấu vân tay cấu hình cloud. Không cache chung ảnh người dùng.
Phần rà dữ liệu chỉ mở trên máy nhóm bằng `npm run start:dev`.
Trong phép thử local sau tối ưu, lượt đầu khoảng 19,5 giây; phiên thứ hai dùng lại mô hình/cache khoảng
1,34 giây. Ba bản nhỏ và metadata cần khoảng 75.748 bytes, không tải ảnh gốc. Các số này thay đổi theo
CPU, ảnh, cache và mạng; không hứa Render sẽ có cùng tốc độ. Khi nạp vision-only, log có thể liệt kê
các trọng số `text_model` là `UNEXPECTED` vì cố ý bỏ nhánh văn bản; thiếu/lệch trọng số cần dùng sẽ bị từ chối.

## 4. Kiểm tra trước khi gửi link cho người ngoài

- Build Linux cài được các phiên bản thư viện đã chốt; cài runtime/kiểm checksum thành công.
- Trang HTTPS, ảnh album, ảnh sai định dạng, nhiều xe và trường hợp ngoài phạm vi đều được thử.
- Camera trên điện thoại được cấp quyền và hoạt động qua HTTPS; không suy từ camera giả lập của test.
- Đối chiếu lấy tối đa ba bản nhỏ từ Supabase; không tải kho ảnh hoặc weights về trình duyệt.
- Không có công cụ nội bộ; lỗi không hiện khóa hoặc stack trace chi tiết.
- Theo dõi RAM và thời gian thực tế khi có vài người dùng; chưa coi hàng đợi inference là kiểm thử tải.
- Chỉ báo độ chính xác mới sau khi nhóm duyệt và chạy bộ ảnh mới theo
  [quy trình đánh giá](independent_evaluation.md).

Hướng dẫn này chưa chứng minh đã triển khai thành công. Tài khoản Render đã kết nối;
chưa tạo service, thêm secret, chọn gói trả phí hoặc tự thay đổi GitHub.
