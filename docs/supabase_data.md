# Dữ liệu ảnh trên Supabase

Project: **CarVision AI**, Singapore, ref `bnfqwqpqxoovoxnwunnx`.
[Dashboard](https://supabase.com/dashboard/project/bnfqwqpqxoovoxnwunnx).

Ảnh gốc nằm trong bucket **private** `carvision-datasets`. Bảng `public.carvision_images`
lưu nhãn, nguồn, giấy phép, thông tin ghi công, SHA-256, đường dẫn object và trạng thái upload.
Object đặt tên theo checksum để tránh tạo thêm bản sao của cùng tệp. Không đưa ảnh hoặc khóa vào Git.
Schema riêng trong `database/cloud_images.sql`, không thay schema thông số SQLite hiện có.

| Dataset | Ảnh lưu trữ | Đề xuất giữ | Mục đích |
|---|---:|---:|---|
| tesla-week3-candidate | 91 | 66 | Nguồn baseline; giữ dấu vết train/validation/test cũ |
| tesla-week4-new-review | 20 | 16 | Đánh giá riêng, không dùng học |
| tesla-supplement-v1 | 137 | 90 | Bổ sung ứng viên, chưa duyệt nhóm xe |

Tổng 248 ảnh gốc, 571.086.707 byte. 76 ảnh đã loại vẫn được giữ để truy vết;
giao diện và lệnh tải chỉ dùng ảnh có `decision=keep` và `uploaded=true`.
Đề xuất giữ không có nghĩa nhãn/nhóm xe đã được xác nhận độc lập.
Các trường `candidate_vehicle_id` và `source_group_hint` là gợi ý, không phải danh tính xe đã duyệt.

Cập nhật 12/11/2026: 82/90 ảnh bổ sung đã dùng trong [thử nghiệm train/validation mở rộng](week4_expanded_training.md).
Kho cloud vẫn giữ bản ứng viên gốc và trạng thái chưa được duyệt độc lập; không coi cả bộ 90 ảnh là test chưa từng thấy.

## Cấu hình trên máy chủ

Không cần thêm thư viện. Dùng môi trường đã cài `requirements-demo.txt`.
Owner cấu hình `CARVISION_SUPABASE_URL` và `CARVISION_SUPABASE_KEY` trong môi trường
chạy Python/Streamlit. `.env.example` chỉ là mẫu tên biến; app không tự đọc `.env`.
Trong PowerShell:

```powershell
$env:CARVISION_SUPABASE_URL = 'https://bnfqwqpqxoovoxnwunnx.supabase.co'
$cloudKey = Read-Host 'Nhap secret key may chu tu Supabase Dashboard' -AsSecureString
$env:CARVISION_SUPABASE_KEY = [System.Net.NetworkCredential]::new('', $cloudKey).Password
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Nhập khóa trực tiếp ở terminal của owner; không gửi vào chat, commit, chia sẻ cho người dùng
hoặc đặt trong JavaScript trình duyệt. Secret/service-role key có quyền cao: chỉ dùng ở máy chủ tin cậy.
Hiện bảng bật RLS, không cấp quyền cho `anon`/`authenticated`; bucket không có policy đọc công khai.
Chưa cấu hình tài khoản/quyền đọc riêng cho từng thành viên. Thành viên có thể xem qua Dashboard
nếu owner cấp quyền project, hoặc dùng bản dữ liệu đã owner tải; không cần chia sẻ server key.

Trong Data Studio chọn **Supabase → Nạp / cập nhật danh mục**. Chọn phiên bản, dòng xe và ảnh.
App chỉ lấy một ảnh đang xem, kiểm tra checksum và giữ một thumbnail trong phiên hiện tại;
không tải toàn bộ dataset xuống máy người xem. Chế độ **Trên máy** vẫn dùng audit chung như trước.

### Ảnh đối chiếu trong chế độ nhận diện · 18/11/2026

Chọn **Nhận diện ảnh → Nguồn ảnh đối chiếu → Supabase riêng tư**. Máy chủ dùng cấu hình ở trên,
truy vấn đúng checksum và lớp của ảnh train đã ghim trong model (`limit=1`), rồi tải bytes qua đường dẫn authenticated.
Giữ nguyên bucket private và RLS; không cấp quyền mới, không gửi server key xuống trình duyệt.
Mỗi lần chỉ hiển thị tối đa ba ảnh; kiểm tra kích thước, SHA-256 và giới hạn pixel trước khi tạo thumbnail.
Nếu thiếu cấu hình hoặc không tải được, kết quả dự đoán vẫn hiển thị và thẻ ảnh báo lỗi; có nút thử tải lại.
Đối chiếu SQL ngày 18/11: 248 ảnh gốc, 172 ảnh keep/uploaded, đủ checksum của 105 ảnh train trong model V3.1.
Đây là kiểm tra metadata; chưa thay thế kiểm thử tải bytes thực tế qua app khi cấu hình khóa máy chủ.

## Tải dữ liệu để chạy audit/huấn luyện

Sau khi cấu hình môi trường, tải từng phiên bản vào một thư mục **mới**:

```powershell
.\.venv\Scripts\python.exe -m src.cloud_data tesla-supplement-v1 data/raw/tesla-supplement-cloud-v1
```

Lệnh kiểm tra kích thước/SHA-256 từng ảnh, giữ nguồn và giấy phép trong `provenance.csv`,
lưu danh mục cloud trong `cloud_manifest.json`, chạy audit trước khi xuất thư mục hoàn chỉnh.
Nếu lỗi, không xuất thư mục dữ liệu dở dang hoặc ghi đè thư mục cũ.
Manifest giữ mục đích và split cũ; công cụ này **không tự tạo lại train/validation/test**.

Để audit trong Data Studio, chọn **Trên máy**, nhập `tesla-supplement-cloud-v1` rồi quét.
Ảnh tải từ cloud vẫn phải qua xác nhận nhãn, rà ảnh gần trùng/nhóm xe và quy trình chia tập trước khi học.
Cloud là nơi quản lý ảnh; huấn luyện vẫn cần đọc ảnh tại máy chạy và inference cần trọng số mô hình.
Chưa có triển khai website nhận diện công khai hoặc tự động học từ ảnh người dùng.

## Vận hành

Không xóa bản local chỉ vì đã upload. Chỉ báo upload hoàn tất sau khi kiểm tra bytes đọc lại và catalog.
Endpoint di chuyển một lần đã được giới hạn token, thời hạn, checksum và danh mục cho phép;
sau khi xác minh đã được khóa (version 2, JWT bật, handler trả 410).
Không dùng endpoint tạm làm API upload ảnh người dùng.
Không chạy lại SQL tạo bảng trên database đã có bảng. Không ghi trực tiếp hàng vào `storage.objects`
để giả lập upload: hàng metadata không thay thế được bytes trong Storage.

Lưu trữ và lượt tải chịu hạn mức gói Supabase; quote tạo project 0 USD không có nghĩa mọi mức sử dụng đều miễn phí.
Không tự bật bucket public, tăng gói hoặc xóa dữ liệu để vượt hạn mức.

Kiểm tra sau upload: đủ 248 object và 248 dòng đã xác minh; tổng byte khớp bản gốc.
Security advisor chỉ có INFO “RLS Enabled No Policy”: đây là cấu hình đóng chủ đích,
chưa cấp quyền người dùng; không tạo policy `true` để làm mất thông báo.
[Giải thích INFO](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy).
Performance advisor không có thông báo. Kiểm thử giao diện dùng ảnh giả;
chưa kiểm tra app đọc cloud bằng khóa máy chủ thực tế vì owner chưa cấu hình khóa.

Tài liệu chính thức: [Storage access control](https://supabase.com/docs/guides/storage/security/access-control),
[phục vụ ảnh riêng tư](https://supabase.com/docs/guides/storage/serving/downloads).
