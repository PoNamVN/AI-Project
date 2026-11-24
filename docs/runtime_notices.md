# Nguồn đi kèm runtime V3.1

Gói nội bộ đầy đủ chứa model CarVision V3.1 đang dùng, hai encoder, detector, đúng ảnh train mà model dùng đối chiếu
và manifest phát triển cũ để công cụ đánh giá kiểm tra chồng lặp.
Không chứa ảnh người dùng, ảnh ứng viên tuần 5, tập đánh giá mới, khóa hoặc toàn bộ kho Supabase.
Việc đóng gói không làm nhãn ảnh được duyệt hoặc tạo kết quả accuracy mới.

Gói `cloud-only` không có ảnh/manifest lịch sử. Gói `compact` cũng không có ảnh; hai encoder đã được
xuất ONNX và nén chỉ trọng số 8 bit, detector xuất dạng float. Đây là biến đổi từ các checkpoint bên dưới,
không phải pretrained checkpoint nguyên bytes; giữ nguồn, giấy phép và ghi chú biến đổi này cùng gói.

- DINOv2: Meta, `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
  [Model card và giấy phép Apache-2.0](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md).
- SigLIP2: Google, `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`.
  [Model card và giấy phép Apache-2.0](https://huggingface.co/google/siglip2-base-patch16-224/blob/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/README.md).
- Detector: Torchvision SSDLite320 MobileNetV3 Large, checkpoint
  `ssdlite320_mobilenet_v3_large_coco-a79551df.pth` từ
  [PyTorch](https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth).
  [Nguồn Torchvision](https://github.com/pytorch/vision) và [tài liệu mô hình](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).
  Bản BSD đi kèm thuộc thư viện Torchvision; không suy ra quyền đối với mọi ảnh COCO từ giấy phép code.

Giữ [Apache-2.0](licenses/Apache-2.0.txt) và [Torchvision BSD](licenses/Torchvision-BSD.txt) cùng gói.
Không sửa trọng số pretrained khi tạo bundle. Ảnh train được chép nguyên bytes; mỗi ảnh giữ
`source_url`, `artist_html`, `license`, `license_url`, `sha256` trong `model.json`.
Mỗi ảnh có điều kiện ghi công/giấy phép riêng, không dùng giấy phép model thay cho giấy phép ảnh.

Gói dành cho máy chạy Python/Streamlit trong nhóm. Người dùng truy cập website chỉ nhận giao diện,
kết quả và tối đa các ảnh đối chiếu đang xem, không tải bộ encoder hay toàn bộ dataset.
