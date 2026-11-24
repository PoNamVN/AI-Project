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

Không thêm thư viện. Cần Node.js 20.12 trở lên và môi trường nhận diện theo hướng dẫn máy mới.
Kết nối plugin Supabase trong Codex dùng để quản lý project; cấu hình app riêng một lần trên máy chủ tin cậy.

1. Owner mở [Settings → API Keys của CarVision AI](https://supabase.com/dashboard/project/bnfqwqpqxoovoxnwunnx/settings/api-keys).
   Lấy secret key bắt đầu bằng `sb_secret_`; nếu chưa có, tạo một key riêng cho máy chủ.
   Không dùng publishable/anon key cho catalog đang đóng. Key từng xuất hiện trong chat/log cần được thu hồi và thay mới.
2. Tại terminal của bạn trong thư mục repo, chạy:

```powershell
npm run setup:cloud
```

3. Dán secret key khi được hỏi; ký tự không hiện ra. Công cụ đọc thử **một ảnh tham chiếu cụ thể**,
   kiểm tra kích thước, checksum và giải mã ảnh trước khi lưu. Không tải cả catalog hoặc nạp weights để kiểm tra kết nối.
4. Khi thành công, cấu hình được lưu vào `.env` đã bị Git bỏ qua. Trên Windows tệp được giới hạn quyền
   cho người đang thiết lập và SYSTEM; trên hệ thống hỗ trợ quyền POSIX tạo tệp với mode 0600.
   Công cụ không ghi đè `.env` đã tồn tại và không lưu cấu hình nếu đọc ảnh thất bại.
5. Khởi động lại app bằng `npm start`. Nếu bản local khác đang giữ cổng 8501, có thể chạy bản cloud riêng:

```powershell
npm start -- --server.port 8502
```

Mở URL terminal cung cấp. Người xem không cần khóa hoặc thao tác chọn nguồn ảnh.
Những lần sau chỉ cần `npm start`; launcher đọc đúng hai tên `CARVISION_SUPABASE_URL` và `CARVISION_SUPABASE_KEY`
từ `.env`, không in giá trị. Cấu hình đã đặt sẵn trong môi trường triển khai được ưu tiên hơn tệp local.
Kiểm tra kết nối thực tế về sau bằng:

```powershell
npm run check:cloud
```

`npm run check` vẫn chỉ kiểm tra môi trường/tệp, không tải ảnh. Khi chạy Python trực tiếp hoặc deploy public,
owner đặt hai biến trong môi trường/secret manager của máy chủ; Python không tự nạp `.env`.
Ví dụ nhập vào phiên PowerShell, không ghi khóa vào câu lệnh:

```powershell
$env:CARVISION_SUPABASE_URL = 'https://bnfqwqpqxoovoxnwunnx.supabase.co'
$cloudKey = Read-Host 'Nhap secret key may chu tu Supabase Dashboard' -AsSecureString
$env:CARVISION_SUPABASE_KEY = [System.Net.NetworkCredential]::new('', $cloudKey).Password
npm start
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

Trong chế độ nhận diện, app tự dùng **Supabase riêng tư** khi máy chủ có đủ hai biến cấu hình ở trên;
người xem không phải chọn nguồn lưu ảnh. Máy chủ dùng cấu hình đó,
truy vấn đúng checksum và lớp của ảnh train đã ghim trong model (`limit=1`), rồi tải bytes qua đường dẫn authenticated.
Giữ nguyên bucket private và RLS; không cấp quyền mới, không gửi server key xuống trình duyệt.
Mỗi lần chỉ hiển thị tối đa ba ảnh; kiểm tra kích thước, SHA-256 và giới hạn pixel trước khi tạo thumbnail.
Nếu thiếu cấu hình hoặc không tải được, kết quả dự đoán vẫn hiển thị và thẻ ảnh báo lỗi; có nút thử tải lại.
Đối chiếu SQL ngày 18/11: 248 ảnh gốc, 172 ảnh keep/uploaded, đủ checksum của 105 ảnh train trong model V3.1.
Đây là kiểm tra metadata; chưa thay thế kiểm thử tải bytes thực tế qua app khi cấu hình khóa máy chủ.

Kiểm tra lại ngày 25/11/2026: project trước đó ở trạng thái INACTIVE, đã yêu cầu khôi phục và xác nhận ACTIVE_HEALTHY.
Catalog vẫn có 248 ảnh uploaded, 172 ảnh keep/uploaded; tìm thấy đủ 248 object tương ứng trong metadata Storage,
không có mục uploaded thiếu object. Đối chiếu checksum và lớp: đủ 105/105 ảnh tham chiếu trong model hiện tại.
Không upload lại, không thay bucket private hoặc RLS. Đây vẫn là kiểm tra metadata, chưa xác minh bytes tải về qua app.
Tiến trình app local tiếp tục dùng ảnh local cho tới khi owner hoàn tất cấu hình và khởi động lại.
Ngày 25/11 đã kiểm tra đọc thực tế một ảnh Cybertruck private qua API: checksum khớp,
giải mã được ảnh 3108 × 2331, 1.759.900 byte. Khóa kiểm tra không được lưu vào dự án;
không coi phép thử CLI này là bằng chứng tiến trình app đã được cấu hình.

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
Performance advisor không có thông báo trong lần kiểm tra trước. Kiểm thử giao diện cloud dùng ảnh giả;
CLI đã đọc một ảnh thật như ghi ở trên, còn tiến trình app cần owner cấu hình và khởi động lại để kiểm tra trực tiếp.

Tài liệu chính thức: [Storage access control](https://supabase.com/docs/guides/storage/security/access-control),
[phục vụ ảnh riêng tư](https://supabase.com/docs/guides/storage/serving/downloads).
# Ảnh đối chiếu thu nhỏ (25/11/2026)

Đã thêm 105 ảnh WEBP thu nhỏ (tối đa 500×320), tổng 2,457,882 bytes so với 248,297,409 bytes ảnh gốc tương ứng.
Ảnh gốc không sửa/xóa. Bản nhỏ nằm ở `previews/v1/<source_sha256>/<preview_sha256>.webp` trong cùng bucket riêng tư.
Metadata/checksum/nhãn liên kết nằm trong `provenance.preview_v1`; giữ nguyên các trường ghi công/nguồn cũ,
không thêm bảng hoặc đổi schema. Chỉ cập nhật liên kết nếu provenance vẫn giống lúc đọc, tránh đè thay đổi của người khác.

Web ưu tiên bản nhỏ đã kiểm SHA-256; nếu chưa có metadata bản nhỏ thì vẫn kiểm và dùng ảnh gốc.
Nếu metadata hoặc bytes bản nhỏ sai, app báo lỗi thay vì âm thầm dùng dữ liệu không khớp.
Ảnh đối chiếu cloud thành công được cache trên máy chủ tối đa 32 ảnh/15 phút, tách theo cấu hình kết nối.
Đây là ảnh tham chiếu từ tập train; không biến chúng thành bộ ảnh đánh giá độc lập.

Lệnh dành riêng chủ máy chủ:

```powershell
npm run prepare:previews
npm run prepare:previews -- --publish
```

Lệnh đầu chỉ chuẩn bị/đếm, lệnh sau mới upload và liên kết. Không ghi đè object đã có; có thể chạy lại để xác minh.
Không dùng lệnh này để gửi ảnh người dùng hoặc ảnh mới chưa được rà nhãn.
