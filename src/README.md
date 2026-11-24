# Các phần code chính

- `data_audit.py`: kiểm tra ảnh, nguồn, nhãn và ảnh trùng; dùng chung cho giao diện và CLI.
- `prepare_dataset.py`, `dataset_manifest.py`: rà và chuẩn bị tập dữ liệu, giữ kiểm tra đường dẫn và chống chồng lặp.
- `train_baseline.py`, `train_transfer.py`: CNN/MobileNetV2, chạy trong môi trường TensorFlow riêng.
- `vision_features.py`, `train_visual.py`, `train_brand_gate.py`: encoder đóng băng và bộ phân loại thử nghiệm V3.1.
- `recognize_vehicle.py`, `vehicle_reasoning.py`: nhận diện, trường hợp chưa xác định và bằng chứng hình ảnh.
- `recognition_workflow.py`: ảnh đối chiếu, lựa chọn chưa xác minh và so sánh hai góc ảnh.
- `cloud_data.py`: truy cập ảnh Supabase riêng tư từ máy chủ, giới hạn kích thước và kiểm tra checksum.
- `check_environment.py`, `evaluate_field.py`: kiểm tra môi trường và đánh giá batch đã duyệt riêng.

Giao diện nằm ở `app/app.py`. Ảnh, cache, model và lựa chọn người dùng không tự trở thành dữ liệu huấn luyện.
