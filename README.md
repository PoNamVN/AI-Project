# CarVision AI — Nhận diện các dòng Tesla

Repo chung: **https://github.com/PoNamVN/AI-Project**.

> Trạng thái 28/10/2026: xác định phạm vi và khảo sát nguồn. Chưa có dataset được duyệt, mô hình đã huấn luyện hay số liệu độ chính xác. App hiện chỉ là trang giới thiệu Streamlit, chưa nhận diện ảnh.

## 1. Giải thích dự án

CarVision AI giúp người không rành xe tìm hiểu chiếc Tesla họ nhìn thấy qua ảnh. Luồng sản phẩm dự kiến:

1. Chụp/tải ảnh xe lên website.
2. AI gợi ý Top-3 dòng Tesla trong phạm vi hỗ trợ.
3. Người dùng đối chiếu ảnh mẫu có quyền sử dụng, xác nhận/chọn lại hoặc chọn **không có mẫu phù hợp**.
4. Bổ sung năm, thị trường và phiên bản nếu biết.
5. Website tra thông số có nguồn và giải thích thuật ngữ dễ hiểu.

Phạm vi đã đổi từ nhiều hãng sang **Tesla-only**. Năm lớp ứng viên: **Model S, Model 3, Model X, Model Y, Cybertruck**. Chỉ chốt sau khi kiểm tra quyền sử dụng, nhãn và số ảnh; đây không phải danh sách xe hiện đang bán. Hãng cố định là Tesla, không phải bài toán phân biệt mọi hãng.

Ảnh không đảm bảo xác định đúng đời, pin, động cơ hay phiên bản. Thiếu thông tin thì để chưa xác định. Xử lý xe không phải Tesla cần được thiết kế và kiểm thử riêng; bộ phân loại năm lớp không tự đảm bảo phát hiện xe ngoài phạm vi.

### Ảnh huấn luyện và database khác nhau thế nào?

| Thành phần | Mục đích | Trạng thái |
|---|---|---|
| Ảnh + nhãn dòng xe | Cho AI học và đánh giá | Chưa có dataset được duyệt |
| Ảnh mẫu | Đối chiếu khi người dùng chọn | Chưa duyệt bộ ảnh và quyền hiển thị |
| SQLite + nguồn thông số | Tra đúng năm/thị trường/phiên bản | Mới có schema dự thảo |
| File mô hình | Dự đoán trong website | Chưa huấn luyện |

`.env` chỉ chứa cấu hình/khóa nếu sau này cần, không chứa toàn bộ database hay ảnh. App giới thiệu hiện không cần API key hoặc `.env`.

Nguồn đang khảo sát, chưa phải quyết định sử dụng:

- [Tesla manuals](https://www.tesla.com/ownersmanual): tham khảo thông số, không phải dataset ảnh.
- [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Tesla_automobiles): kiểm tra nhãn, giấy phép và ghi công từng file.
- [Roboflow community Tesla candidate](https://universe.roboflow.com/mark-l7omf/tesla-375su-2ciui): chưa xác minh truy cập, giấy phép hay chất lượng; không phải dữ liệu chính thức của Tesla.

Ảnh công khai trên web không mặc định được dùng để huấn luyện hoặc đăng lại.

## 2. Thành viên và nhiệm vụ đề xuất

| Thành viên | Phụ trách |
|---|---|
| Phan Lê Phương Nam | Điều phối, phạm vi, câu hỏi nghiên cứu, báo cáo |
| Trần Võ Bá Vương | Nguồn ảnh, quyền sử dụng, nhãn và chất lượng |
| Lê Văn Công Nguyên | Tiền xử lý, mô hình, chia tập và đánh giá |
| Lê Hoàng Triều | Website, Git và tích hợp |
| Trần Tiến Đạt | Nguồn thông số, phiên bản và giải thích thuật ngữ |

Ghi nhiệm vụ, bằng chứng và giờ thực tế. Phân công không có nghĩa đã hoàn thành.

## 3. Cài công cụ

Hướng dẫn chính dành cho **Windows + VS Code + PowerShell**.

1. Cài [Git](https://git-scm.com/downloads).
2. Cài [Python 3.12](https://www.python.org/downloads/); thêm Python vào PATH nếu trình cài có tùy chọn. GPU/thư viện huấn luyện sẽ xác minh riêng.
3. Cài [Visual Studio Code](https://code.visualstudio.com/).
4. Trong Extensions, cài **Python** của Microsoft; thêm **Jupyter** khi làm notebook.
5. Chấp nhận lời mời cộng tác GitHub của Nam nếu cần push. Public chỉ cấp quyền đọc, không tự cấp quyền ghi.

Mở terminal mới:

```powershell
git --version
python --version
```

Nếu `python` mở Microsoft Store, thử `py -3.12 --version` hoặc sửa cài đặt/PATH. Có thể dùng `py -3.12` thay `python` ở bước tạo môi trường.

## 4. Clone và mở trong VS Code

Ví dụ tạo thư mục Projects trong tài khoản Windows:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Projects" | Out-Null
Set-Location "$env:USERPROFILE\Projects"
git clone https://github.com/PoNamVN/AI-Project.git
Set-Location AI-Project
code .
```

Nếu thiếu lệnh `code`, mở VS Code → **File → Open Folder** → chọn `AI-Project`.

**Chỉ clone một lần.** Lần sau mở lại thư mục và pull. Không clone lồng trong repo và không `git init` lại.

Trong VS Code chọn **Terminal → New Terminal**:

```powershell
Get-Location
Get-ChildItem
git remote -v
```

Bạn phải đứng ở thư mục có `README.md`, `requirements-demo.txt`, `app/`. Remote origin cần trỏ tới `PoNamVN/AI-Project`, không phải repo cũ của bạn khác.

## 5. Tạo môi trường và cài thư viện

Tại thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Trong VS Code: **Ctrl+Shift+P → Python: Select Interpreter → Enter interpreter path**, chọn `AI-Project\.venv\Scripts\python.exe`.

Dùng trực tiếp Python trong `.venv` nên không cần kích hoạt hay đổi Execution Policy. Nếu muốn kích hoạt, chạy `.\.venv\Scripts\Activate.ps1`; khi bị chặn cứ dùng cách gọi trực tiếp ở trên.

## 6. Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Mở **Local URL** terminal hiển thị, thường là `http://localhost:8501`. Nếu trình duyệt không tự mở, dán URL vào trình duyệt.

**Kết quả đúng hiện tại:** trang CarVision AI hiển thị trạng thái chuẩn bị dữ liệu và luồng Tesla dự kiến. Chưa có chức năng nhận diện thật. Chạy được trang không có nghĩa AI đã hoạt động.

- Dừng: **Ctrl+C** trong terminal đang chạy.
- Sửa app rồi lưu; làm mới trình duyệt nếu cần.
- Nếu cổng bận:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py --server.port 8502
```

Những lần sau chỉ mở project và chạy lệnh Streamlit, không tạo lại môi trường.

### macOS/Linux

Sau khi clone và vào repo:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python -m streamlit run app/app.py
```

Hướng dẫn macOS/Linux chưa được thử trong phiên này.

## 7. Cấu trúc và cách dùng file

```text
AI-Project/
  README.md                 Hướng dẫn nhóm và phạm vi dự án
  .gitignore                Loại file riêng/tạm khỏi Git
  requirements-demo.txt     Thư viện chạy trang giới thiệu
  requirements.txt          Thư viện ML dự kiến, chưa khóa phiên bản
  app/app.py                Điểm chạy Streamlit
  data/README.md            Quy tắc nguồn ảnh và chia tập
  data/raw/                 Ảnh/nhãn gốc, hiện trống
  data/processed/           Kết quả xử lý, hiện trống
  database/README.md        Giải thích database
  database/schema.sql       Schema SQLite dự thảo
  models/README.md          Quy tắc lưu model, chưa có trọng số
```

Bản công khai ban đầu là khung tối thiểu trên. Tài liệu cá nhân/giảng viên, repo lồng và slide tạm không đi kèm.

Khi nguồn được duyệt, đặt ảnh gốc vào `data/raw/`, đầu ra vào `data/processed/`, ghi nguồn, nhãn và giấy phép. Quy ước ảnh chi tiết sẽ chốt theo dataset. Không coi tên file là nhãn đúng; không để ảnh trùng/cùng xe rải sang train và test.

`database/schema.sql` chưa có bản ghi xe. App hiện chưa tạo/kết nối database. Không cần MySQL/MongoDB để chạy trang giới thiệu; `sqlite3` có sẵn trong Python.

## 8. Phần AI làm tiếp thế nào?

Thí nghiệm dự kiến: CNN cơ sở; MobileNetV2 học chuyển giao; cùng MobileNetV2 có tăng cường ảnh lúc học. Dùng cùng lớp và cách chia tập để so sánh.

1. Xác minh nguồn và quyền ảnh.
2. Kiểm tra ảnh hỏng, nhãn thiếu/sai, trùng và số ảnh mỗi lớp.
3. Chốt lớp và danh sách train/validation/test theo nhóm ảnh cùng xe.
4. Chạy thử nhỏ rồi huấn luyện và đánh giá.
5. Tích hợp mô hình, ảnh mẫu và thông số có nguồn vào website.

`requirements.txt` là danh sách ML **dự kiến**, chưa phải môi trường đã kiểm chứng/khóa phiên bản. Chỉ mở trang giới thiệu thì dùng `requirements-demo.txt`. Khi làm ML, thống nhất Python, framework, phần cứng và lưu phiên bản. Danh sách hiện theo TensorFlow; không trộn PyTorch từ repo khác khi chưa thống nhất. Colab là tùy chọn.

## 9. Quy trình Git cho thành viên

### Khai báo danh tính lần đầu

Thay thông tin ví dụ bằng của bạn; có thể dùng email riêng tư GitHub cung cấp:

```powershell
git config --global user.name "Ho Ten Cua Ban"
git config --global user.email "email-cua-ban"
```

### Bắt đầu công việc

Kiểm tra `git status`. Nếu có thay đổi chưa commit, hoàn tất trên nhánh đang làm trước khi chuyển.

```powershell
git status
git switch main
git pull --ff-only origin main
git switch -c nam/ten-cong-viec
```

Ví dụ nhánh khác: `vuong/tesla-data-audit`, `nguyen/baseline-cnn`, `trieu/website-flow`, `dat/tesla-specs`. Một nhánh làm một việc nhỏ. Nhánh đã có thì dùng `git switch ten-nhanh`.

### Commit và push

Ví dụ chỉ sửa `app/app.py` trên nhánh `nam/ten-cong-viec`:

```powershell
git diff
git add app/app.py
git diff --cached
git commit -m "docs: clarify Tesla demo status"
git push -u origin nam/ten-cong-viec
```

Thay tên nhánh đúng của bạn. Chọn từng file, kiểm tra phần đã stage; tránh gửi cả dữ liệu bằng `git add .`.

Tạo **Pull Request** vào `main` trên GitHub, mô tả đã sửa gì/kiểm tra thế nào, nhờ một người xem trước khi merge. Đây là quy ước nhóm, chưa khẳng định branch protection đã bật. Không force-push main.

Nếu có conflict hoặc non-fast-forward, dừng để xem nguyên nhân; không tự chạy `reset --hard`, xóa `.git` hay force-push. Push không được: kiểm tra tài khoản, quyền cộng tác, remote; có thể đăng nhập qua Git Credential Manager. Không ghi token/mật khẩu vào code.

## 10. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Không nhận Python/Git | Cài công cụ, kiểm tra PATH, mở lại terminal/VS Code |
| Không tìm thấy app/requirements | Vào thư mục gốc repo |
| No module named streamlit | Cài requirements-demo bằng Python trong .venv |
| VS Code dùng nhầm Python | Chọn interpreter .venv của repo |
| Activate bị PowerShell chặn | Gọi trực tiếp .venv/Scripts/python.exe, không cần đổi chính sách |
| Cổng 8501 bận | Dừng app cũ hoặc chọn 8502 |
| Clone được nhưng push không được | Nhận lời mời và đăng nhập đúng tài khoản có quyền |
| Không có ảnh/model/database sau clone | Artifact chưa chuẩn bị/phân phối, không phải Git tải thiếu |
| App chỉ hiện thông báo | Đúng trạng thái bản giới thiệu hiện tại |

## 11. Không đưa lên Git

- `.env`, token, mật khẩu, khóa và dữ liệu cá nhân.
- `.venv`, cache, log và thư mục tạm.
- Ảnh dataset/người dùng, file nén dữ liệu.
- Trọng số model, database chứa dữ liệu.
- Tài liệu giảng viên chưa có quyền chia sẻ.

`.gitignore` không tự bỏ file đã commit. Nếu lộ bí mật, báo nhóm và thu hồi khóa; chỉ xóa file ở commit mới chưa đủ.

## 12. Kiểm tra bản khởi đầu

Đã thử cài requirements-demo.txt trên Windows với Python 3.12.14 và Streamlit 1.66.1, kiểm tra render app và HTTP health thành công. Schema SQLite tạo được bốn bảng trong database tạm. Chưa kiểm thử bộ ML, GPU hoặc hướng dẫn macOS/Linux.

## 13. Quy tắc báo cáo

Phân biệt kế hoạch với kết quả đã chạy. Chưa có audit, độ chính xác hoặc giờ thực tế thì ghi chưa đo/chưa ghi nhận. Kiểm tra lựa chọn người dùng trước khi dùng làm nhãn học. Không hứa nhận diện mọi xe khi chưa có bằng chứng.
