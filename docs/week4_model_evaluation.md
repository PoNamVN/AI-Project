# Tuần 4 · Phát triển và đánh giá mô hình

## Cập nhật mô hình 224 × 224 và fine-tuning

Bản `week4-transfer224-v1` dùng lại đúng 66 ảnh và cách chia cũ, nhưng đọc ảnh gốc đúng SHA-256 để tạo ảnh RGB 224 × 224. Trọng số ImageNet tương ứng 224 × 224 được nạp lại. Lần cải thiện này thay đồng thời độ phân giải, loss cân bằng lớp và cách học hai giai đoạn, nên chưa tách được đóng góp riêng của từng thay đổi.

Loss khi train có trọng số lớp theo công thức `w_c = N / (K × n_c)` và `L_i = -w_c × log(p_c)`. Ở đây `N=45`, `K=5`, `n_c` là số ảnh train của dòng c. Cybertruck có trọng số 1,0; Model 3 và Y là 0,75; Model S là 1,2857; Model X là 1,8. Cách tính chỉ dùng train. Validation vẫn dùng cross-entropy không trọng số để so sánh với các lần trước.

Giai đoạn đầu học Dense head với learning rate 0,001, tối đa 25 epoch. Sau đó mở 30 tầng cuối backbone, giữ BatchNormalization đóng băng và gọi backbone với `training=False`. Fine-tuning dùng Adam 0,00001 tối đa 15 epoch. Cả hai giai đoạn đều dừng sớm theo validation, patience 5. Chọn trọng số có loss validation thấp nhất qua hai giai đoạn trước khi đo test.

Lần chạy thực tế gồm 16 epoch head và 6 epoch fine-tuning. Trọng số được chọn ở epoch đầu giai đoạn fine-tuning, tức epoch 17 toàn lần chạy, với loss validation 0,9773. Trên 14 ảnh test đã biết, mô hình đúng 7/14 (50,0%), macro-F1 47,1%, Top-3 đúng 12/14 (85,7%). So với CNN, accuracy ngang nhau, macro-F1 tăng từ 25,6% lên 47,1%. Model S đúng 1/1, Model 3 đúng 1/2, Model Y đúng 4/5, Cybertruck đúng 1/4, Model X đúng 0/2.

### Kiểm tra trên ảnh mới

Sau khi chốt mô hình, lấy 20 ảnh mới từ Wikimedia, lọc tác giả và hash đã có trong bộ cũ, rồi rà ảnh trước khi dự đoán. Giữ 16 ảnh, loại 4 vì chủ thể không phù hợp, ảnh prototype hoặc có nguy cơ trùng xe đã thấy. Số ảnh giữ gồm S 4, Model 3 hai, X 4, Y 4 và Cybertruck hai. Chưa xác minh độc lập tuyệt đối về xe và nhãn. Bộ mới dùng riêng để đánh giá, chưa đưa vào train hoặc chọn cấu hình.

| Mô hình | Accuracy trên 16 ảnh mới | Macro-F1 | Top-3 |
|---|---|---|---|
| CNN cũ | 1/16 · 6,25% | 2,35% | 56,25% |
| Mô hình cải thiện | 4/16 · 25,0% | 23,49% | 68,75% |

Cải thiện tương đối đã được quan sát, nhưng điểm ảnh mới cho thấy khả năng nhận diện còn yếu. Bộ kiểm tra nhỏ từ một nguồn và nhãn rà bằng hình ảnh chưa đại diện cho ảnh đường phố bất kỳ. Giữ kết quả này để công khai hạn chế trước khi tiếp tục bổ sung dữ liệu. Nhu cầu chính tiếp theo là thêm các xe độc lập có nhãn được người phụ trách xác nhận, nhất là Model S và X.

### Chạy bản cải thiện và xem ba gợi ý

```powershell
.\.venv-ml\Scripts\python.exe -m src.train_transfer data/processed/tesla-week3-v1 --originals data/raw/tesla-week3-candidate --output data/processed/week4-transfer224-v1
.\.venv-ml\Scripts\python.exe -m src.predict_vehicle data/processed/week4-transfer224-v1 --image "duong-dan-den-anh.jpg"
```

Đổi tên đầu ra nếu thư mục đã có. `model.keras` là mô hình được chọn. `metrics.json` lưu seed, trọng số lớp, lịch sử hai giai đoạn và các chỉ số. Lệnh dự đoán trả ba dòng ứng viên cùng điểm xếp hạng chưa hiệu chỉnh xác suất. Người dùng cần đối chiếu ảnh, và xe ngoài phạm vi vẫn có thể bị gán một nhãn Tesla.

## Thiết kế thí nghiệm

Giữ bộ 66 ảnh và cách chia tuần 3, gồm 45 ảnh train, 7 validation và 14 test. Hai thí nghiệm tuần 4 dùng cùng seed 42, learning rate 0,001, batch size 8 và giới hạn 25 epoch. Chọn trọng số theo loss validation thấp nhất, dừng sớm sau 5 epoch không cải thiện. Giữ bộ chia và trọng số CNN tuần 3 để đối chiếu.

| Thí nghiệm | Cách học |
|---|---|
| CNN tuần 3 | Học từ đầu, ba tầng tích chập |
| MobileNetV2 | Trọng số ImageNet, đóng băng backbone, học lớp Dense năm nhãn |
| MobileNetV2 + tăng cường ảnh | Cùng backbone, thêm lật ngang, xoay ±10,8 độ và zoom nhẹ khi train |

Ảnh RGB 128 × 128 được đưa qua lớp Rescaling từ [0, 255] về [-1, 1], rồi MobileNetV2, GlobalAveragePooling và Dense softmax. Backbone có BatchNormalization và chạy ở chế độ suy luận để giữ thống kê đã học. Mô hình có tổng 2.264.389 tham số, chỉ 6.405 tham số được cập nhật. Tăng cường ảnh nằm trong model và tự tắt khi validation, test hoặc predict. Ảnh gốc và danh sách chia được giữ riêng.

Loss là sparse categorical cross-entropy vì mỗi ảnh có một nhãn dòng xe dạng số nguyên. Adam tối ưu trọng số lớp phân loại. Cấu hình này chọn theo kế hoạch chuyển giao trên bộ nhỏ, trước khi đọc kết quả hai lần chạy tuần 4.

## Chạy lại trong VS Code

Dùng Python 3.12 và môi trường đã cài `requirements-baseline.txt` theo hướng dẫn tuần 3. Lần đầu cần mạng để tải trọng số ImageNet từ máy chủ TensorFlow. Lệnh mặc định lưu trọng số tải về trong thư mục cache Keras của tài khoản.

```powershell
.\.venv-ml\Scripts\python.exe -m src.train_baseline data/processed/tesla-week3-v1 --output data/processed/week4-mobilenet-v1 --architecture mobilenetv2 --seed 42 --epochs 25
.\.venv-ml\Scripts\python.exe -m src.train_baseline data/processed/tesla-week3-v1 --output data/processed/week4-mobilenet-aug-v1 --architecture mobilenetv2 --augment --seed 42 --epochs 25
.\.venv-ml\Scripts\python.exe -m src.compare_models data/processed/week3-baseline-v1 data/processed/week4-mobilenet-v1 data/processed/week4-mobilenet-aug-v1 --output data/processed/week4-comparison-v1
```

Nếu tên đầu ra đã có, chọn tên mới và dùng đúng tên đó trong lệnh so sánh. Mỗi run chứa `baseline.keras` (tên tệp dùng chung cho các kiến trúc), `metrics.json` và `test_predictions.csv`. Thư mục so sánh chứa `comparison.csv`, `per_class.csv`, `errors.csv` và `selection.json`. Công cụ so sánh yêu cầu cùng mã kiểm tra danh sách chia, thứ tự nhãn và danh sách ảnh test.

## Kết quả ngày 11/11/2026

| Mô hình | Epoch chạy / chọn | Loss validation thấp nhất | Đúng trên test | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| CNN tuần 3 | 15 / 10 | 1,5837 | 7/14 | 50,0% | 25,6% |
| MobileNetV2 | 11 / 6 | 1,5442 | 6/14 | 42,9% | 23,3% |
| MobileNetV2 + tăng cường ảnh | 10 / 5 | 1,5887 | 3/14 | 21,4% | 13,3% |

Theo loss validation, MobileNetV2 thường được chọn làm ứng viên để khảo sát tiếp. Trên tập test đã cố định, CNN tuần 3 có điểm cao hơn cả hai phương án tuần 4. Tăng cường ảnh trong cấu hình này làm điểm test giảm. Các kết quả này chỉ mô tả lần chạy hiện tại.

### Kết quả theo dòng

| Dòng | CNN | MobileNetV2 | MobileNetV2 + tăng cường |
|---|---|---|---|
| Cybertruck | 0/4 | 2/4 | 3/4 |
| Model 3 | 2/2 | 0/2 | 0/2 |
| Model S | 0/1 | 0/1 | 0/1 |
| Model X | 0/2 | 0/2 | 0/2 |
| Model Y | 5/5 | 4/5 | 0/5 |

### Khó khăn huấn luyện và phân tích lỗi

MobileNetV2 thường đạt 100% accuracy train từ epoch 8 trong khi validation chỉ 3–4/7 ảnh đúng. Khoảng cách này phù hợp với nguy cơ quá khớp trên bộ ảnh nhỏ. Mô hình được khôi phục về epoch 6 theo loss validation. Bản tăng cường ảnh đạt 4/7 validation ở trọng số được chọn nhưng dự đoán cả 5 ảnh Model Y trong test thành Model X.

Rà bảng ảnh test cho thấy ảnh Model S có xe đạp và nền phố gần xe, Model X có góc thân xe dễ gần với Model Y, còn nhiều ảnh Model Y có góc chụp từ trên xuống. Đây là ngữ cảnh để đặt giả thuyết về ảnh hưởng phông nền và góc chụp. Cần thử nghiệm kiểm soát mới xác định được nguyên nhân. Lật ngang có thể tạo chữ và biển số đảo chiều, còn zoom/xoay thay đổi khung xe. Tác động riêng của từng phép biến đổi chưa được đo.

Tập test có 14 ảnh và đã được xem từ tuần 3, nên đây là bộ so sánh phát triển đã biết. Một ảnh đúng thêm làm accuracy thay đổi khoảng 7,1 điểm phần trăm. Việc MobileNetV2 có cải thiện một số ảnh Cybertruck chưa chứng minh khả năng khái quát cao hơn. Mỗi thí nghiệm mới có một seed và nhãn/nhóm xe vẫn là ước tính.

## Hướng cải thiện và bàn giao

1. Bổ sung các xe độc lập, nhiều góc chụp, ưu tiên Model S và Model X, rồi rà nhãn và nhóm xe. Nếu đổi dữ liệu, tạo phiên bản chia mới và chạy lại các mô hình trên cùng bản mới.
2. Tạo bộ đánh giá đường phố độc lập để dùng sau khi chọn cấu hình bằng validation. Bổ sung ảnh ngoài phạm vi khi nghiên cứu khả năng từ chối xe không hỗ trợ.
3. Thử riêng từng phép tăng cường và cân nhắc fine-tuning ở learning rate nhỏ sau khi có thêm dữ liệu. Chốt cấu hình trước khi đánh giá độc lập.

Đã có hai mô hình tuần 4, cấu hình, lịch sử học, chỉ số, ma trận nhầm lẫn và bảng dự đoán sai. Nam đang phụ trách triển khai kỹ thuật. Nhóm cần xác nhận lịch bàn giao, người trình bày, giờ dự kiến/thực tế và cách xử lý chênh lệch trước khi nộp báo cáo.

Tài liệu kỹ thuật tham khảo gồm [MobileNetV2 và thang đầu vào](https://keras.io/api/applications/mobilenet/mobilenet_models/) và [học chuyển giao, đóng băng backbone](https://keras.io/guides/transfer_learning/).
