# Hướng dẫn cho công cụ AI trong CarVision AI

Đọc `README.md`, `CONTRIBUTING.md` và code liên quan trước khi sửa. Các quy tắc này áp dụng toàn repo, trừ hướng dẫn cụ thể hơn trong thư mục con.

- Dự án hiện chỉ khảo sát dữ liệu Tesla; chưa có dataset được duyệt hoặc mô hình nhận diện đã huấn luyện. Không tạo kết quả hoặc số liệu để làm đầy giao diện/báo cáo.
- Chỉ sửa phần cần cho yêu cầu. Giữ thay đổi chưa commit của người dùng và file ngoài nhiệm vụ; không tự dọn hoặc ghi đè chúng.
- Tái sử dụng `src/data_audit.py` cho logic kiểm tra dữ liệu; giao diện nằm ở `app/app.py`. Không nhân đôi logic kiểm tra trong giao diện.
- Giữ kiểm tra đầu vào, giới hạn đọc ảnh, bảo vệ đường dẫn và xử lý lỗi. Không ghi đè dữ liệu trong `data/raw/`; đầu ra phải nằm riêng.
- Không thêm dữ liệu thật, khóa, `.env`, trọng số hoặc database dữ liệu vào commit. Không đọc/in giá trị bí mật để debug.
- Không tự thay framework, phạm vi lớp xe, schema hay thêm phụ thuộc ngoài yêu cầu. Nếu cần thay, giải thích ảnh hưởng và cập nhật tài liệu/cấu hình liên quan.
- Thay đổi logic cần kiểm thử phù hợp. Chạy `python -m unittest discover -s tests -p "test_*.py" -v` bằng môi trường đã cài `requirements-demo.txt`; kiểm tra giao diện nếu có thay đổi hiển thị. Chỉ sửa tài liệu thì đọc lại diff và kiểm tra liên kết/lệnh liên quan.
- Không commit/push/merge khi người dùng chưa yêu cầu. Khi được yêu cầu công bố code, dùng nhánh riêng và PR; giữ nguyên ruleset, không force-push vào nhánh chung.
- Khi báo kết quả, nêu rõ đã sửa gì, đã kiểm tra gì và giới hạn còn lại. Không nói đã chạy test, huấn luyện hay cập nhật GitHub nếu chưa có bằng chứng.
