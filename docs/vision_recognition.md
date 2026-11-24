# Nhận diện Tesla V3 và giải thích bằng ảnh

## Đã triển khai

Giữ các model TensorFlow cũ. V3 dùng môi trường PyTorch riêng, không đổi 5 lớp Tesla.
**Bản mặc định hiện tại là V3.1**, thư mục `vision-recognition-v3-brand`, bổ sung head Tesla/xe khác được học có giám sát.
Ảnh tải lên đi qua detector ô tô COCO, cắt vùng xe có thêm viền, rồi trích đặc trưng từ DINOv2-small và SigLIP2-base.
Bộ phân loại ridge cân bằng lớp chỉ học từ 105 ảnh train; backbone giữ nguyên. 29 ảnh validation dùng chọn cấu hình.

Thử 5 cách dùng đặc trưng, mỗi cách 3 hệ số ridge (0,01 / 0,1 / 1), cùng một baseline zero-shot.
Quy tắc chọn trước khi đọc điểm kiểm tra: macro-F1 validation cao nhất, rồi accuracy, rồi loss thấp nhất.
Kết quả chọn **combined_crop-ridge-0.1**. Không thử sửa cấu hình theo hai bộ kiểm tra 14/16 ảnh.

Trên validation, cấu hình tốt nhất của DINO toàn ảnh đúng 21/29; DINO crop 20/29;
SigLIP toàn ảnh 25/29; SigLIP crop 24/29; kết hợp crop **26/29**.
Vì vậy không khẳng định crop riêng lẻ luôn làm điểm cao hơn. Lợi ích chính trong đợt này là đặc trưng pretrained mạnh hơn và cách kết hợp được chọn bằng validation.

## Kết quả đo ngày 12/11/2026

Giữ nguyên 105 train / 29 validation / 14 ảnh kiểm tra phát triển; 16 ảnh riêng không tham gia fit hoặc chọn cấu hình.

| Chỉ số | MobileNetV2 mở rộng V2 | Bộ phân loại V3 |
|---|---:|---:|
| Validation Top-1 | 13/29 — 44,8% | 26/29 — 89,7% |
| Validation macro-F1 | 0,4690 | 0,8977 |
| 14 ảnh cũ Top-1 | 6/14 — 42,9% | 14/14 — 100% |
| 16 ảnh riêng Top-1 | 5/16 — 31,3% | 15/16 — 93,8% |
| 16 ảnh riêng Top-3 | 13/16 — 81,3% | 16/16 — 100% |

**Bộ phân loại** là bài toán: biết ảnh thuộc một trong 5 dòng Tesla, xếp hạng dòng nào.
Các tỷ lệ trên không phải độ chính xác xác nhận hãng hoặc hiệu quả toàn bộ sản phẩm.

Với detector, head hãng V3.1 và quy tắc chọn xe, trên 14 ảnh có 13 gợi ý Top-1 đúng, 1 ảnh giữ lại;
trên 16 ảnh có **13 gợi ý Top-1 đúng / 16 ảnh — 81,3%**, 2 ảnh chưa tìm thấy vùng xe và 1 ảnh cần người dùng chọn xe.
Đếm ảnh bị giữ lại là chưa giải quyết được, không loại chúng khỏi mẫu số để làm đẹp tỷ lệ.
Gợi ý được trả về vẫn có thể mang trạng thái **chưa chắc hãng Tesla** hoặc **chưa chắc dòng xe**;
không được trình bày như kết luận đã xác nhận. Trên 14 ảnh có một gợi ý còn chưa chắc dòng xe.
Top-3 theo luồng thực tế là 13/16. Số ảnh trả gợi ý không được thay mẫu số để quảng cáo accuracy.

Gate ban đầu chỉ dùng tương đồng SigLIP2 với mô tả, từ chối 6/8 ảnh ngoài Tesla và còn 2 trường hợp nhận nhầm.
Do cơ chế này yếu, V3.1 giữ nguyên bộ phân loại dòng xe và học thêm ridge nhị phân từ **105 Tesla + 24 xe khác**.
24 ảnh ngoại thất được giữ từ 32 ứng viên thuộc các danh mục mẫu xe cụ thể; loại ảnh nội thất, chi tiết và cảnh nhiều xe.
Ảnh học mới có tác giả khác bộ ảnh âm validation/kiểm tra theo metadata đã chuẩn hóa.
Thử 3 hệ số ridge, giữ ngưỡng margin cố định 0 và yêu cầu giữ ít nhất 90% Tesla validation;
chọn bằng macro-F1 trên **29 Tesla + 9 xe khác validation**. Cấu hình chọn hệ số 1: đúng 37/38, giữ 29/29 Tesla, từ chối 8/9 xe khác.

Sau khi đóng băng head hãng, chạy lại luồng inference: **8/8 ảnh ngoài Tesla được từ chối**, không trả dòng Tesla.
Đây là bộ rất nhỏ, chỉ có một số xe thường/xe điện; chưa phải bộ kiểm tra open-world đầy đủ.
Bộ 8 ảnh đã được quan sát ở giai đoạn chẩn đoán gate cũ nên kết quả V3.1 là kiểm tra phát triển, không phải holdout mới hoàn toàn.
Không điều chỉnh hệ số theo điểm của 8 ảnh này. Chưa kiểm chứng được với mọi xe BYD/Rivian hoặc mọi vật thể khác.

## Người dùng nhìn thấy gì?

1. Chọn ảnh JPG/PNG/WEBP từ album/máy hoặc chụp bằng camera trình duyệt, tối đa 30 MB và 25 triệu pixel. App không tự lưu ảnh vào dataset/Supabase.
2. Nếu không tìm thấy ô tô rõ, đề nghị chụp lại gần hơn, đủ sáng hoặc cắt bớt cảnh nền.
3. Nếu có hai xe có vùng ảnh lớn tương đương, app đánh số khung và yêu cầu chọn xe.
4. Khi thiếu căn cứ về hãng, app giữ lại hoặc ghi rõ “Chưa chắc hãng Tesla; nếu là Tesla…”.
5. Trả Top-3 và điểm xếp hạng. Điểm này được biến đổi từ classifier để xếp hạng, **không phải xác suất đúng đã hiệu chuẩn**.
6. Đối chiếu ba ảnh **train** gần nhất trong không gian đặc trưng, kèm nhãn nguồn, tác giả và giấy phép.
7. Nếu bật giải thích chi tiết, che lần lượt 9 vùng và đo thay đổi điểm ứng viên. Vùng đỏ là vùng che làm điểm giảm, không tự gán nó thành logo/đèn/cửa.
8. Hiện các giả thuyết ảnh–văn bản về thân xe/góc nhìn khi các mô tả đủ khác nhau. Kiểu cửa luôn yêu cầu kiểm chứng thủ công; không suy ra cửa cánh chim từ việc model đoán Model X.

### Luồng demo tuần 5 · 18/11/2026

**Bổ sung đối chiếu hai góc ảnh:** sau kết quả đầu, tải ảnh thứ hai và xác nhận hai ảnh chụp cùng chiếc xe,
rồi bấm **Đối chiếu hai góc ảnh**. Hai ảnh được phân tích độc lập; file trùng bị từ chối.
Nếu hai Top-1 khác nhau, hệ thống yêu cầu đối chiếu thêm. Nếu một ảnh chưa chắc hãng/không thấy xe/có nhiều xe,
không chốt kết quả chung. Ngay cả hai ảnh cùng gợi ý cũng không phải xác nhận hoặc xác suất tin cậy mới.
Không hợp nhất điểm hoặc ghi đè dự đoán ảnh đầu; JSON bổ sung kết quả ảnh hai và thông báo đối chiếu.
Phần kỹ thuật nằm trong mục mở rộng để màn hình chính dễ đọc hơn.
Quy trình đánh giá ảnh mới ở [independent_evaluation.md](independent_evaluation.md).

Mở `npm start`, mặc định vào **Nhận diện ảnh**. Các chế độ audit local và danh mục Supabase vẫn có ở sidebar.
Sau khi phân tích, mỗi ứng viên Top-3 có **một ảnh train cùng lớp gần nhất** để đối chiếu.
`candidate_references` bổ sung các ảnh này; `references` vẫn giữ ba ảnh train gần nhất toàn bộ cho bằng chứng trước đây, có thể trùng lớp.
Không đổi head, trọng số, cách tính điểm hoặc bộ kiểm tra; việc bổ sung luồng này không tạo accuracy mới.

Bấm **Chọn Model…** dưới ảnh gợi ý hoặc **Không có mẫu phù hợp**.
Để chọn một trong năm dòng khác, mở **Chọn dòng khác hoặc tải lại ảnh tham khảo**, chọn dòng rồi bấm **Lưu lựa chọn của tôi**.
JSON có thêm `user_review`, checksum ảnh đầu vào và manifest mô hình. Lựa chọn không ghi đè dự đoán AI,
không phải nhãn đã xác minh, không ghi database, không lưu ảnh người dùng hoặc tự đưa vào huấn luyện.
Ảnh mới, nguồn chụp mới hoặc phân tích lại sẽ xóa kết quả/lựa chọn cũ. Tải JSON nếu muốn giữ kết quả trước khi đóng phiên.

Mặc định phân tích nhanh, không tạo bản đồ che vùng. Muốn xem bản đồ, mở **Tùy chọn phân tích**,
bật giải thích chi tiết và phân tích lại. Điểm xếp hạng nằm trong phần kỹ thuật, không phải xác suất đúng.

Ảnh đối chiếu tự lấy từ **Supabase riêng tư** khi máy chủ có đủ cấu hình kết nối; app truy vấn
checksum/nhãn của từng ảnh và chỉ tải tối đa ba ảnh cần hiển thị. Khi chưa cấu hình, bản local dùng tệp trên máy đã kiểm tra checksum.
Người xem không chọn nguồn lưu ảnh; xem cấu hình máy chủ trong [supabase_data.md](supabase_data.md).
Cache chỉ giữ ba thumbnail trong phiên; nút tải lại cho phép thử lại khi kết nối lỗi. Không cần toàn bộ ảnh raw trên máy chủ
nếu dùng cloud, nhưng vẫn cần model, pretrained weights và train feature vectors trên máy chủ.

Camera cần trình duyệt cấp quyền; localhost dùng được trên máy hỗ trợ camera, truy cập từ điện thoại qua mạng cần HTTPS.
Ảnh HEIC không nằm trong các định dạng upload hỗ trợ hiện tại; xuất JPEG/PNG trước khi tải nếu cần.

Các mô tả bộ phận không phải chuỗi suy luận của bộ phân loại. Chưa có bộ phát hiện logo, đèn, cửa hoặc đời xe chuyên biệt.
Năm, phiên bản và thông số để `null`; không bịa thông số từ ảnh.
Ảnh train gần giống không chứng minh ảnh người dùng là cùng xe hoặc cùng đời xe.

## Cài và chạy trên Windows / VS Code

Tạo môi trường riêng để giữ nguyên môi trường TensorFlow:

```powershell
python -m venv .venv-vision
.\.venv-vision\Scripts\python.exe -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv-vision\Scripts\python.exe -m pip install -r requirements-demo.txt -r requirements-vision.txt
.\.venv-vision\Scripts\python.exe -m streamlit run app/app.py
```

Chọn **Nhận diện ảnh** ở sidebar. Máy hiện tại đã có môi trường tương ứng tại
`.kickoff-build/vision-v3/Scripts/python.exe`, có thể dùng ngay:

```powershell
.\.kickoff-build\vision-v3\Scripts\python.exe -m streamlit run app/app.py
```

Hoặc từ thư mục gốc repo, nếu đã có Node.js/npm và môi trường trên:

```powershell
npm start
```

Không cần `npm install`. Lệnh dùng `.venv-vision` nếu có, rồi môi trường `.kickoff-build/vision-v3`, cuối cùng `.venv`; model và cache vẫn cần được chuẩn bị như hướng dẫn.

Model mặc định: `data/processed/vision-recognition-v3-brand/`.
`CARVISION_VISION_RUN` có thể chọn một thư mục model khác trong `data/processed`, không chấp nhận đường dẫn ngoài khu vực đó.
Vision không cần API key. Ảnh chỉ được xử lý trên máy chủ app; inference không gửi ảnh tới OpenAI/Hugging Face.
Lần chuẩn bị đầu cần tải weights công khai vào cache riêng. Chạy thường dùng `local_files_only=True`, không tự tải model trong UI.
Máy clone repo mới phải có artifacts và cache hoặc chạy lại bước chuẩn bị; Git không chứa weights/dataset.

Thử CLI và lưu ảnh bằng chứng vào thư mục mới:

```powershell
.\.venv-vision\Scripts\python.exe -m src.recognize_vehicle data/processed/vision-recognition-v3-brand --image data/raw/tesla-week4-new-review/model_s/37533990.jpg --output data/processed/recognition-example-new
```

Với ảnh nhiều xe, dùng `--vehicle-index 0` hoặc chỉ số trong kết quả detection. Không ghi đè thư mục báo cáo đã tồn tại.

## Tái lập thí nghiệm

Đầu vào cần đúng các bộ nguồn gốc đã được kiểm tra, `tesla-expanded-v2/splits.csv`
và `non-tesla-gate-review-v1/review.csv` (17 ảnh đề xuất giữ từ 26 ứng viên).
Các ảnh âm rộng cảnh/đồ chơi/bộ phận đã loại được giữ riêng để truy vết, không đưa vào điểm gate.

```powershell
.\.venv-vision\Scripts\python.exe -m src.train_visual --output data/processed/vision-recognition-reproduction --download
.\.venv-vision\Scripts\python.exe -m src.train_brand_gate data/processed/vision-recognition-reproduction --output data/processed/vision-recognition-brand-reproduction
```

Tham số `--download` cho phép tải checkpoint đã pin revision; bỏ tham số khi cache đã sẵn sàng.
Head hãng cần `non-tesla-brand-training-v3/review.csv` cùng đúng ảnh nguồn. Ảnh âm train/validation/check không trùng checksum.
Script kiểm tra đường dẫn, checksum, giữ nguyên tập cũ, lưu lựa chọn model trước khi trích đặc trưng bộ kiểm tra.
Đầu ra được ghi tạm và chỉ đổi thành thư mục đích khi xong. Không ghi đè model cũ.
Cache embeddings giảm chi phí chạy lại; chúng chỉ hợp lệ với phiên bản detector/preprocessing/checkpoint hiện tại.

Artifacts private local:

- `vision-recognition-v3/model.json`: cấu hình, lựa chọn validation, ngưỡng và thông tin ghi công ảnh train.
- `vision-recognition-v3/classifier.npz`: classifier và embeddings train/text, không chứa ảnh người dùng.
- `vision-recognition-v3/evaluation.json` và CSV: dự đoán, số ảnh từ chối, kết quả từng lớp.
- `vision-recognition-v3-brand/`: cùng bộ phân loại dòng xe, thêm weights hãng; `runtime_evaluation.json`/CSV ghi kết quả đúng theo luồng giao diện cuối.
- `vision-v3-example-s/`: ví dụ giai đoạn gate văn bản ban đầu.
- `vision-v31-example-s/`: ví dụ V3.1 thật, ảnh khoanh vùng, heatmap, JSON giải thích và kiểm tra giao diện.
- `.kickoff-build/vision-v3-environment.txt`: phiên bản môi trường đã dùng; cache weights ở `.kickoff-build/`.

## Kiểm tra và giới hạn

Đã chạy unit test đường dẫn, kích thước ảnh, lớp dữ liệu, tính toán ridge/ranking,
từ chối ảnh ngoài phạm vi, lựa chọn nhiều xe và hợp đồng giải thích không khẳng định bộ phận.
Đã chạy AppTest với model thật: Model S đúng, có ba ảnh train tham chiếu, heatmap 9 vùng và không có lỗi giao diện.
Ở lần đo V3.1 trên CPU máy hiện tại: nạp model khoảng **12,6 giây**; khi đã nạp, dự đoán khoảng **0,8 giây**,
và dự đoán kèm giải thích 9 vùng khoảng **4,4 giây** trên một ảnh mẫu. Đây không phải benchmark cho mọi máy.
Model được cache trên server; inference chỉ tính nhánh đặc trưng được chọn, không tính thêm toàn ảnh khi không cần.
Đã đối chiếu embeddings tối ưu với cache thí nghiệm (sai số cho phép 1e-5); không đổi lớp dự đoán.
Heatmap dùng ảnh làm việc tối đa 672×672 để giới hạn bộ nhớ.

Cả hai bộ Tesla 14/16 ảnh đều đã xem điểm từ các thí nghiệm trước; pretrained có thể đã gặp ảnh web liên quan.
Không có bằng chứng hai bộ này tách khỏi dữ liệu pretraining. Chưa có bộ ảnh đường phố mới được giữ kín.
Nhãn/nhóm xe vẫn là ứng viên, còn cần xác nhận độc lập. Không công bố 93,8% như độ chính xác ngoài thực tế.
Gate ngoài Tesla chưa được chứng minh trên tập lớn và có thể sai/bỏ sót Tesla; cần thêm negative khó, ảnh đường phố và kiểm thử độc lập trước triển khai công khai.

Tài liệu mô hình: [DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2),
[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224),
[SSDLite COCO](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).
# Tối ưu vận hành ngày 25/11/2026

Recognizer chỉ nạp nhánh ảnh của checkpoint SigLIP2 cố định; nhánh mã hóa văn bản không dùng lúc suy luận
vì classifier đã chứa các vector văn bản. Không đổi trọng số, ngưỡng hoặc cách tính điểm. Engine dành cho
huấn luyện/tạo đặc trưng vẫn mặc định đầy đủ. Đối chiếu năm ảnh train cũ cho kết quả và đặc trưng giống hệt
bản đầy đủ (sai khác tối đa 0); không coi đây là bộ đánh giá mới.

Ảnh đối chiếu lấy bản nhỏ đã kiểm checksum từ Supabase, cache chung có giới hạn; ảnh người dùng không cache chung.
Hai đường phân tích ảnh được xếp lần lượt trong một process để giảm nguy cơ nhiều inference làm tăng RAM cùng lúc.
Mặc định tắt công cụ rà dữ liệu; mở riêng trên máy nhóm bằng `npm run start:dev`.
Xem [chuẩn bị Render](deploy_render.md) để phân biệt kết quả kiểm tra local với triển khai công khai.
