# Tuần 5 — Demo CarVision AI và bàn giao

Cập nhật 18/11/2026. Phạm vi năm dòng Tesla. V3.1 và các chỉ số tuần 4 được giữ nguyên;
đợt này hoàn thiện luồng sử dụng, không huấn luyện lại hoặc tạo số liệu accuracy mới.

## Chạy và thử

Từ thư mục repo đã chuẩn bị môi trường/model/cache:

```powershell
npm run check
npm start
```

`npm run check` báo thư viện, file model, cache encoder/detector và trạng thái cấu hình cloud (không in khóa).
Đây là kiểm tra sự hiện diện tệp, chưa chứng minh mọi weights đọc được hoặc inference chạy đúng.
Nếu máy khác báo thiếu, chuẩn bị theo [hướng dẫn nhận diện](vision_recognition.md) trước khi quay.

Mở Local URL terminal cung cấp. Trang mặc định là **Nhận diện ảnh**.

1. **Chọn từ album / máy**: dùng JPG/JPEG/PNG/WEBP, một xe đủ sáng và trọn thân.
2. Hoặc **Chụp trực tiếp**: cấp quyền camera khi muốn dùng; camera mạng cần HTTPS, localhost có thể dùng trên máy.
3. Bấm **Phân tích ảnh**. Nếu ảnh có nhiều xe đáng chú ý, chọn số khung rồi phân tích xe đã chọn.
4. Đọc Top-3; mỗi dòng có một ảnh train cùng lớp để đối chiếu và nguồn/giấy phép.
5. Chọn dòng xe hoặc **Không có mẫu phù hợp / chưa xác định**, bấm **Lưu lựa chọn của tôi**.
6. Đọc bằng chứng tương đồng và bản đồ che vùng. Chúng không xác nhận logo/đèn/cửa, đời hoặc phiên bản.
7. **Tải giải thích · JSON** để giữ dự đoán, lựa chọn chưa xác minh, checksum ảnh và manifest model.

Có thể thêm **Ảnh góc thứ hai**, xác nhận cùng một chiếc xe và bấm **Đối chiếu hai góc ảnh**.
Kết quả cho biết hai gợi ý thống nhất, mâu thuẫn hoặc thiếu căn cứ; không tăng điểm hay xác nhận tự động.
Nếu góc thứ hai có nhiều xe, dùng ảnh chỉ có chiếc muốn đối chiếu; chưa có bộ chọn xe riêng cho góc thứ hai.
Phần giải thích chuyên sâu ở mục mở rộng. Xem [quy trình kiểm tra ảnh mới](independent_evaluation.md).

Ảnh và lựa chọn không tự ghi database hoặc trở thành dữ liệu huấn luyện. Đổi ảnh/nguồn chụp hoặc phân tích lại
xóa kết quả và lựa chọn cũ. Lựa chọn người dùng không làm thay đổi Top-3 hoặc tỷ lệ chính xác đã đo.

## Kho ảnh riêng

Mặc định dùng **Trên máy chủ**. Nếu server đã cấu hình theo [supabase_data.md](supabase_data.md),
chọn **Supabase riêng tư**. App truy vấn theo checksum ghim trong model, không tải cả catalog/dataset;
mỗi phiên chỉ cache tối đa ba thumbnail. Nếu gặp lỗi, dùng **Tải lại ảnh đối chiếu** hoặc chọn lại nguồn local.
Server vẫn cần model, pretrained weights và vector train; người xem không tải chúng.
Máy chủ hiện chưa được cấu hình biến môi trường cloud trong lần kiểm tra 18/11; không đưa secret key vào chat hay client.

## Các tình huống cần quay thật

| Tình huống | Nội dung cần chỉ ra |
|---|---|
| Một Tesla rõ ràng | Top-3, ảnh từng ứng viên, nguồn và giải thích |
| Chọn lại dòng hoặc không phù hợp | Lựa chọn riêng của người dùng, dự đoán AI vẫn giữ nguyên |
| Đổi sang ảnh khác | Kết quả/lựa chọn cũ được xóa |
| Ảnh nhiều xe | Chọn chiếc muốn nhận diện theo số khung |
| Xe ngoài phạm vi hoặc ảnh không thấy xe | Thông báo giới hạn thực tế; không chọn một kết quả đẹp để thay cho đánh giá |
| Camera nếu thiết bị hỗ trợ | Xin quyền, chụp và chạy cùng luồng phân tích |

Chuẩn bị ảnh có quyền dùng và thử trước khi quay; lưu cả trường hợp thất bại để giải thích. Không cam kết
mọi ảnh ngoài Tesla đều bị từ chối. Điểm xếp hạng không phải xác suất đúng đã hiệu chuẩn.

## Video có tất cả thành viên

Thầy yêu cầu: “record a project demo video featuring all team members and submit the video file”.
Đề xuất mỗi người xuất hiện và nói, có tên trên màn hình; quay màn hình công cụ và ghép thành MP4.
Chưa có thời lượng, độ phân giải, dung lượng tối đa hoặc hạn nộp được xác nhận; không coi đề xuất là yêu cầu thầy.

| Người | Phần nói đề xuất |
|---|---|
| Nam | Bài toán, phạm vi và luồng sử dụng |
| Vương | Dữ liệu, nhãn, chia tập, nguồn và quyền ảnh |
| Nguyên | DINOv2/SigLIP2 đóng băng, ridge và pipeline |
| Triều | Demo công cụ, các tình huống khó và kết quả |
| Đạt | Giới hạn, phần chưa hoàn thiện và hướng phát triển |

Quay thao tác thật; kiểm tra phát lại file cuối cùng từ đầu đến cuối. Khi nêu hiệu năng, phân biệt
classifier 15/16 với toàn luồng V3.1 13/16 trên batch phát triển đã xem; chưa có street holdout độc lập.

## Bàn giao và giới hạn

- README, requirements-demo/vision, hướng dẫn model/cache và dữ liệu riêng cần đủ để máy khác chạy lại.
- Ảnh/trọng số/khóa không vào Git. Nhóm nhận code qua nhánh/PR sau khi Nam yêu cầu công bố.
- Chưa có tra cứu thông số/năm/trim có nguồn, đăng nhập, lưu phản hồi hoặc triển khai public trong đợt này.
- Khả năng tải ảnh tham chiếu cloud cần cấu hình server và kiểm thử bytes thực tế; không bật bucket public để demo.
- Camera vật lý và các trình duyệt điện thoại cần kiểm tra trên thiết bị trước khi quay.
- Chốt ngày bàn giao, người thực hiện và giờ dự kiến/thực tế từ nhật ký, không tự điền.

Kiểm thử:

```powershell
.\.kickoff-build\vision-v3\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Môi trường khác thay bằng Python đã cài requirements-demo và requirements-vision. Test có TensorFlow
sẽ bỏ qua nếu chưa cài requirements-baseline; ghi rõ trong báo cáo kiểm thử.
