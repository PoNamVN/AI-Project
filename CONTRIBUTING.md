# Quy định làm việc nhóm — CarVision AI

Áp dụng cho mọi thành viên và mọi thay đổi do công cụ AI hỗ trợ. Mục tiêu là giữ bản dùng chung chạy được, bảo toàn dữ liệu và có thể kiểm tra lại kết quả.

## 1. Làm đúng phạm vi

- Nhận nhiệm vụ và báo nhóm trước khi bắt đầu để tránh hai người sửa cùng phần.
- Mỗi nhánh, mỗi PR tập trung vào một việc. Không sửa/xóa phần của người khác chỉ để tiện làm việc của mình.
- Giữ phạm vi Tesla-only theo README. Thay đổi lớp xe, framework, cấu trúc thư mục, schema database hoặc thư viện phải nêu lý do và ảnh hưởng trong PR để Nam duyệt trước khi gộp.
- Không thay toàn bộ file bằng code AI khi chưa đọc và hiểu phần bị thay. Người gửi PR chịu trách nhiệm kiểm tra code được tạo.

## 2. Bảo vệ Git và công việc của nhau

- Cập nhật `main` bằng `git pull --ff-only origin main`, rồi tạo nhánh riêng. Quy trình chi tiết nằm trong [README](README.md#9-quy-trình-git-cho-thành-viên).
- Chỉ push nhánh của mình; không push thẳng hoặc tự merge vào `main`. Nam (`PoNamVN`) kiểm tra và merge PR.
- Không force-push nhánh chung, xóa nhánh của người khác, tắt ruleset hoặc thay quyền cộng tác để vượt lỗi Git.
- Trước khi chuyển nhánh hoặc sửa conflict, kiểm tra `git status` và lưu công việc đang làm. Không dùng `reset --hard`, `clean -fd` hay xóa `.git` để chữa lỗi khi chưa xác định rõ dữ liệu sẽ mất và có bản sao cần thiết.
- Khi conflict chưa rõ nguyên nhân, giữ nguyên file và trao đổi với người phụ trách; không chọn toàn bộ “ours/theirs” chỉ để hết báo lỗi.
- Chỉ stage các file thuộc nhiệm vụ, đọc `git diff --cached` trước khi commit. Không dùng `git add .` khi chưa kiểm tra toàn bộ thay đổi.

## 3. Bảo vệ dữ liệu và kết quả nghiên cứu

- Không sửa hoặc ghi đè ảnh gốc trong `data/raw/`. Đầu ra xử lý đặt riêng; giữ nguồn, giấy phép, nhãn và mã xe nếu có.
- Không commit ảnh dataset/người dùng, trọng số model, database dữ liệu hoặc báo cáo có thông tin riêng. Chia sẻ qua nơi lưu nhóm đã thống nhất và có quyền truy cập phù hợp.
- Chỉ sử dụng ảnh khi đã kiểm tra quyền sử dụng. Nhãn thư mục và nhãn do AI gợi ý cần được kiểm tra, không mặc định là sự thật.
- Giữ ảnh trùng/cùng một xe trong cùng tập dữ liệu. Sau khi chốt train/validation/test, không đổi cách chia để làm đẹp điểm.
- Dùng validation để chọn mô hình/tham số; không điều chỉnh theo test. Nếu đã xem test rồi thay mô hình, ghi rõ việc đó khi báo cáo.
- Ghi phiên bản dữ liệu, cấu hình, cách chia và kết quả thực tế của thí nghiệm. Không tự điền accuracy, số ảnh hoặc giờ làm chưa đo.

## 4. Bảo mật và môi trường chạy

- Không đưa token, mật khẩu, `.env`, thông tin cá nhân hoặc tài liệu nội bộ vào Git, ảnh chụp màn hình hay log công khai. Nếu cần cấu hình mẫu, chỉ dùng giá trị giả.
- Nếu lộ khóa, báo Nam và thu hồi/đổi khóa ngay; xóa file khỏi commit mới không làm khóa cũ an toàn trở lại.
- Cài thư viện trong `.venv`; cập nhật file requirements phù hợp khi đổi phụ thuộc. Không bắt cả nhóm đổi thiết lập máy chỉ để chạy được trên máy mình.
- Không bỏ kiểm tra đường dẫn, giới hạn ảnh, kiểm tra đầu vào hoặc xử lý lỗi để demo chạy nhanh hơn.

## 5. Trước khi gửi PR

1. Đọc lại diff, bỏ file tạm và thay đổi ngoài nhiệm vụ.
2. Nếu sửa code audit/giao diện, chạy tại thư mục gốc:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
   ```

3. Nếu sửa giao diện, chạy app và kiểm tra phần vừa sửa; đính kèm ảnh nếu giúp người duyệt hiểu. Nếu sửa tài liệu, kiểm tra nội dung và lệnh hướng dẫn; không cần chạy lại test code chỉ vì sửa chữ.
4. Trong PR ghi đã đổi gì, cách kiểm tra, kết quả và phần chưa kiểm tra. Không xóa hoặc bỏ qua test lỗi để đủ điều kiện merge.
5. Chờ Nam duyệt; sửa tiếp trên cùng nhánh nếu được góp ý. Sau merge, cập nhật `main` trước nhiệm vụ tiếp theo.

## Khi phát hiện lỗi trên bản chung

Báo nhóm kèm cách tái hiện và commit liên quan, tránh đưa dữ liệu riêng vào mô tả. Đề xuất bản sửa nhỏ qua PR; nếu cần hoàn tác, Nam thực hiện bằng commit revert để giữ lịch sử. Không sửa lịch sử nhánh chung để che lỗi.

Các quy định trong file là quy trình nhóm, không tự khóa quyền GitHub. Quyền push/merge phải được bảo vệ bằng ruleset; mẫu PR cũng không tự xác nhận rằng test đã chạy.
