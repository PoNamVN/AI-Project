# Kiểm tra trước khi công bố tuần 4–5

Ngày kiểm tra: **25/11/2026**, Windows, Python 3.12.14. Đây là kết quả chuẩn bị bản phát hành,
không phải xác nhận đã commit/push/merge hoặc đánh giá accuracy mới.

## 1. Phạm vi file

Danh sách cụ thể trong [scripts/release-files.txt](../scripts/release-files.txt), gồm 70 file được chọn
(cả những file nền đã có trên main), không phải 70 file đều mới thay đổi.
Phạm vi: app nhận diện, code dùng chung, kiểm thử, requirements, lệnh khởi động, công cụ tạo bundle,
schema catalog ảnh và tài liệu cần để nhóm chạy lại.

Không chọn tài liệu prompt cá nhân, các bản Canva/slide tạm, notebook khung, báo cáo cũ,
`weppy-project-sync`, dữ liệu thật, trọng số, môi trường Python hoặc khóa. Các file này vẫn giữ nguyên trên máy.
Chỉ `.env.example` với giá trị mẫu được chọn; `.env` thật không được đọc hoặc đưa vào gói code.
Đã rà các mẫu token phổ biến trong file dự kiến công bố, không có khớp; đây không phải kiểm toán bảo mật toàn diện.
`.gitignore` bổ sung môi trường vision và các định dạng trọng số để giảm nguy cơ đưa nhầm lên Git.

## 2. Kiểm thử thực tế

| Kiểm tra | Kết quả |
|---|---|
| Bộ Python trong môi trường vision mới | 36 đạt, 4 TensorFlow được bỏ qua có chủ đích |
| Chạy riêng 4 kiểm thử TensorFlow | 4/4 đạt trong môi trường tuần 3 có TensorFlow 2.20.0 |
| Lệnh Node khởi động | 1/1 đạt |
| Dependency trong môi trường mới | 65 package, kiểm tra tương thích đạt |
| `npm run check` trong bản sao | Có model/cache/thư viện; cloud là tùy chọn, chưa cấu hình |
| Khởi động đích của `npm start` | HTTP health `ok`, trang web HTTP 200; đã dừng server thử |
| Chạy app với weights thật | Có Top-3, ba ảnh đối chiếu, lưu lựa chọn chưa xác minh và nút JSON; không có lỗi AppTest |
| Checksum gói runtime cuối | ZIP và toàn bộ 125 file được kiểm tra khớp |

Đã phát hiện và sửa kiểm thử `validate-only` phụ thuộc manifest trên máy Nam.
Kiểm thử hiện tự tạo fixture trong thư mục tạm, vẫn kiểm tra từ chối khi thiếu lịch sử phát triển;
không bỏ hoặc nới kiểm tra chống trùng của code đánh giá.
Công cụ đóng gói có kiểm thử whitelist, checksum, đường dẫn không hợp lệ, liên kết, thiếu manifest,
không ghi đè và lỗi ghi không công bố ZIP dở dang.

Lệnh đã dùng trên máy Nam:

```powershell
.\.kickoff-build\vision-v3\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.kickoff-build\week3-ml\Scripts\python.exe -m unittest discover -s tests -p "test_train_baseline.py" -v
node --test tests/test_start.cjs
```

Trong bản sao đã cài riêng `.venv-vision`, cũng chạy toàn bộ bộ Python trên.
Không ghép TensorFlow và PyTorch vào một môi trường chỉ để tránh dòng `skipped`.

## 3. Bản sao và runtime

Tạo clone cục bộ từ main nền `e3c85b4`, sau đó chỉ áp các file trong danh sách phát hành.
Vì thay đổi chưa commit nên đây là bản sao kiểm tra release candidate, không phải nhánh đã công bố trên GitHub.
Cài lại thư viện vào môi trường mới với `include-system-site-packages = false`; không dùng site-packages của môi trường cũ.
Việc cài dùng uv từ requirements và kho wheel chính thức; người trong nhóm có thể dùng pip theo [hướng dẫn máy mới](team_setup.md).
Không kiểm tra macOS/Linux hoặc một máy vật lý khác.

Gói cuối trên máy Nam: `.kickoff-build/handoff-v2/carvision-runtime-v31.zip` cùng file `.zip.sha256`.
Dung lượng **1.886.843.158 byte**, gồm:

- Hai file classifier/model của V3.1 và cache pretrained đúng revision.
- 105 ảnh train nguyên bytes dùng hiển thị tham chiếu, kèm nguồn/ghi công/giấy phép trong model metadata.
- Năm manifest lịch sử dạng review/splits để máy mới kiểm tra ảnh đã từng dùng, không có ảnh kiểm tra đi kèm.
- Thông tin nguồn, giấy phép và manifest checksum của các file trong gói.

SHA-256 ZIP:

```text
964796596cee48ffe854511af78bf2840ee9f20fa56f69858e438e70d5cd8a29
```

Model và cache trong gói cuối khớp các file đã dùng chạy app trong bản sao.
Ảnh chạy thử là ảnh train cũ `tesla-week3-candidate/model_s/117846425.jpg`, trả gợi ý Model S.
Test chặn kết nối TCP ra ngoài, giữ loopback cho Streamlit và bật Hugging Face offline;
không dùng ảnh này để báo độ chính xác hoặc coi là kiểm tra độc lập.

Bundle nằm ngoài Git và chưa upload lên nơi chia sẻ của nhóm.
Nam cần gửi riêng ZIP và checksum cho thành viên muốn chạy local; người chỉ xem website không tải bundle.

## Còn nằm ngoài phạm vi kiểm tra này

Camera vật lý/điện thoại, tải bytes Supabase qua app với khóa máy chủ thật, triển khai public,
nhãn được người rà độc lập, accuracy mới và video đủ thành viên chưa được hoàn tất trong lần này.
29 ảnh ứng viên tuần 5 vẫn chờ người rà; không huấn luyện lại, đổi ngưỡng hoặc sửa số liệu lịch sử.

Code có thể được đưa sang nhánh riêng và mở PR để duyệt với các giới hạn trên.
Không push thẳng main hoặc thay ruleset; lần chuẩn bị này chưa thực hiện commit/push/merge.
