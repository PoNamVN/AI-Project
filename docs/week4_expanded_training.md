# Thử nghiệm bổ sung ảnh Tesla — 12/11/2026

Đã huấn luyện một bản MobileNetV2 mới với dữ liệu bổ sung. Bản mở rộng có validation loss thấp hơn,
Top-3 tốt hơn trên hai bộ kiểm tra cũ, nhưng Top-1 vẫn thấp và giảm trên bộ 14 ảnh.
Đây là kết quả thử nghiệm cục bộ, chưa đủ kết luận nhận diện tốt xe ngoài đường.

## Rà soát dữ liệu

- Rà lại contact sheet 137 ảnh và mở lớn các ảnh cần xem kỹ; nhãn vẫn là ứng viên, chưa được người thứ hai xác nhận.
- Tính dHash/pHash trên 248 ảnh gốc để tìm ảnh gần trùng. Cặp duy nhất vượt ngưỡng là hai ảnh Model S đã bị loại từ đợt đầu.
- Ngưỡng tìm ứng viên: pHash 63 bit có khoảng cách <=10 hoặc dHash 64 bit <=6. Phương pháp này không phát hiện được mọi ảnh cùng xe/crop/đổi góc.
- Từ 90 ảnh đề xuất giữ, tạm loại 8 ảnh có nguồn tác giả liên quan dữ liệu trước; giữ 82 ảnh cho thử nghiệm mở rộng.
- Gom nguồn theo tên/liên kết tác giả, gom thêm một số alias một cách thận trọng; tạo 55 nhóm nguồn mới. Đây là nhóm proxy, không phải 55 danh tính xe đã xác nhận.
- Các ảnh bổ sung cùng nhóm nguồn luôn nằm cùng train hoặc validation; giữ nguyên 45/7/14 ảnh và nhóm xe của phiên bản cũ.
- Không đưa 16 ảnh đánh giá riêng vào quá trình học hoặc validation. Giữ nguyên bản gốc và model cũ.

## Bộ dữ liệu và cách học

Phiên bản `data/processed/tesla-expanded-v2`: **148 ảnh**, gồm **105 train / 29 validation / 14 kiểm tra cũ**.
Tăng train từ 45 lên 105 ảnh. 82 ảnh mới gồm 60 train và 22 validation.

| Dòng xe | Train | Validation | Kiểm tra cũ |
|---|---:|---:|---:|
| Cybertruck | 18 | 5 | 4 |
| Model 3 | 19 | 8 | 2 |
| Model S | 22 | 4 | 1 |
| Model X | 26 | 6 | 2 |
| Model Y | 20 | 6 | 5 |

Giữ MobileNetV2 ImageNet, ảnh 224×224, class weights theo số ảnh train, seed 42 và batch 8.
Học head với LR 0,001 tối đa 25 epoch; thử fine-tuning 30 lớp cuối trừ BatchNorm với LR 0,00001 tối đa 15 epoch.
Early stopping patience 5 theo validation loss, khôi phục checkpoint tốt nhất. Chạy thực tế 24 epoch head và 7 epoch fine-tuning.
Chọn **head epoch 19** vì loss thấp hơn checkpoint fine-tuning. Không đổi cấu hình sau khi xem điểm bộ kiểm tra.

Manifest mới SHA-256: `35fa47a3fe2695733d5c5e743c0cff7c87b767e04d543a2c256fc84ed97e6ca4`.
Manifest gốc giữ nguyên SHA-256: `b86bc51129b362bd6e226a6b4bf6e7ef70860f4bec4f43122b3126faec8fb1f1`.

## So sánh thực tế

Chạy lại cả hai model trên **cùng 29 ảnh validation**; quy tắc chọn đã chốt là loss thấp nhất.
Lưu quyết định trong `selection.json` trước khi chạy script so sánh hai bộ kiểm tra.

| Chỉ số validation | Model trước | Model mở rộng |
|---|---:|---:|
| Loss | 1,3410 | 1,0823 |
| Top-1 | 11/29 — 37,9% | 13/29 — 44,8% |
| Macro-F1 | 0,4066 | 0,4690 |
| Top-3 | 25/29 — 86,2% | 25/29 — 86,2% |

| Bộ kiểm tra | Model trước Top-1 | Mở rộng Top-1 | Model trước Top-3 | Mở rộng Top-3 |
|---|---:|---:|---:|---:|
| 14 ảnh cũ | 7/14 — 50% | 6/14 — 42,9% | 12/14 — 85,7% | 13/14 — 92,9% |
| 16 ảnh riêng | 4/16 — 25% | 5/16 — 31,3% | 11/16 — 68,8% | 13/16 — 81,3% |

Top-3 chỉ nghĩa nhãn đúng nằm trong ba gợi ý, vẫn cần người dùng đối chiếu và xác nhận.
Trên 16 ảnh, Model X và Model Y đều chưa có dự đoán Top-1 đúng ở bản mở rộng; cần ghi nhận điểm yếu này.
Không công bố accuracy 100% trên train như hiệu quả ngoài thực tế: khoảng cách train–validation cho thấy overfitting.

## Chạy và kiểm tra lại

Sau khi có đúng ảnh gốc, manifest và môi trường `requirements-baseline.txt`, có thể tái chạy vào thư mục mới:

```powershell
.\.venv-ml\Scripts\python.exe -m src.train_transfer data/processed/tesla-expanded-v2 --originals data/raw --output data/processed/transfer-expanded-reproduction
```

Thử một ảnh bằng model mới:

```powershell
.\.venv-ml\Scripts\python.exe -m src.predict_vehicle data/processed/week4-transfer-expanded-v2 --image data/raw/tesla-week4-new-review/model_s/37533990.jpg
```

Artifacts local (không nằm trong Git):

- `data/processed/tesla-expansion-review-v2/`: quyết định từng ảnh, nhóm nguồn, contact sheet cặp gần trùng và bản gán tập trước chuẩn hóa.
- `data/processed/tesla-expanded-v2/`: ảnh chuẩn hóa, `splits.csv`, thống kê và ghi chú dữ liệu.
- `data/processed/week4-transfer-expanded-v2/`: model, lịch sử loss, checkpoint và dự đoán 14 ảnh.
- `data/processed/week4-expansion-comparison-v2/`: model selection, metrics, checksum, dự đoán từng ảnh và kiểm tra CLI.

Đã kiểm tra SHA-256 ảnh gốc, manifest gốc không đổi, không trùng checksum trong bộ mở rộng,
nhóm xe/proxy không nằm ở nhiều split, không có ảnh của bộ 16 trong train/validation và model cũ không bị ghi đè.
Script chẩn đoán/chuẩn bị/so sánh tại `.kickoff-build/` dùng assertions để kiểm tra các điều kiện này.

## Giới hạn và bước tiếp theo

Cả hai bộ 14/16 ảnh đều đã được xem điểm trước đó. Chúng là bộ kiểm tra phát triển, không phải test độc lập mới.
90 ảnh bổ sung đã được chẩn đoán trước; 82 ảnh nay tham gia train/validation nên không được dùng điểm trên toàn bộ 90 ảnh làm bằng chứng tổng quát hóa.
Nhãn và nhóm xe còn cần xác nhận độc lập; chỉ chạy một seed, chưa đo từ chối xe ngoài Tesla, đời hoặc phiên bản.

Bước tiếp theo là bổ sung ảnh đa dạng có nhãn/nhóm xe được xác nhận, đặc biệt Model 3/S/X/Y,
và thu một holdout mới chưa xem điểm. Chốt cấu hình bằng validation trước khi mở holdout đó.
Chưa đổi model mặc định của một ứng dụng triển khai, chưa cập nhật slide/Google Docs và chưa commit/push.
