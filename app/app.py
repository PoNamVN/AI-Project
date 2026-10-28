"""CarVision Week 2: local, read-only Tesla dataset exploration."""

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_audit import MODELS, audit_dataset, report_csv


st.set_page_config(page_title="CarVision · Data Studio", page_icon="🚘", layout="wide")
st.markdown("""
<style>
.stApp {background: #f3f6fa; color: #102633;}
.block-container {max-width: 1280px; padding-top: 2.5rem;}
h1,h2,h3 {letter-spacing: -.035em;}
.cv-hero {background: #091d2a; border-left: 5px solid #62c5de;
padding: 2.2rem 2.5rem; border-radius: 12px; margin-bottom: 1.8rem;}
.cv-kicker {color: #75d0e3; font-size: .78rem; letter-spacing: .16em; font-weight: 700;}
.cv-hero h1 {color: #f4f8fb; font-size: 2.8rem; margin: .35rem 0;}
.cv-hero p {color: #c0d2dd; margin-bottom: 0; max-width: 760px;}
[data-testid="stMetric"] {background: white; padding: 1rem 1.2rem;
border: 1px solid #dce5ed; border-radius: 10px;}
[data-testid="stSidebar"] {background: #e9eff5;}
</style>
<div class="cv-hero">
<div class="cv-kicker">CARVISION AI / WEEK 02</div>
<h1>Tesla Data Studio</h1>
<p>Khám phá ảnh, kiểm tra chất lượng và chuẩn bị bằng chứng trước khi huấn luyện.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Nguồn dữ liệu")
    folder = st.text_input("Thư mục bên trong data/raw", value="tesla", help="Ví dụ tesla hoặc một thư mục con đã có ảnh.")
    scan = st.button("Quét dữ liệu", type="primary", width="stretch")
    st.caption("Chỉ đọc ảnh trên máy. Không tải ảnh lên dịch vụ ngoài và không sửa ảnh gốc.")
    st.divider()
    st.markdown("**5 lớp ứng viên**")
    st.write("Model S · Model 3 · Model X · Model Y · Cybertruck")
    st.caption("Nhãn thư mục cần được người hiểu xe kiểm tra. Ứng dụng chưa nhận diện xe.")

if scan:
    st.session_state.pop("audit", None)
    raw = (ROOT / "data/raw").resolve()
    dataset = (raw / folder).resolve()
    try:
        if not dataset.is_relative_to(raw):
            raise ValueError("Chọn một thư mục nằm bên trong data/raw.")
        with st.spinner("Đang đọc ảnh và kiểm tra dữ liệu…"):
            report = audit_dataset(dataset)
        st.session_state.audit = (str(dataset), report)
    except (ValueError, OSError) as error:
        st.error(str(error))

saved = st.session_state.get("audit")
if saved is None:
    st.subheader("Bắt đầu bằng ảnh thật")
    st.write("Đặt ảnh vào thư mục theo dòng xe, sau đó bấm **Quét dữ liệu**. Khi chưa có dataset, chưa có thống kê hay kết quả mô hình.")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.code("data/raw/tesla/\n  model_s/\n  model_3/\n  model_x/\n  model_y/\n  cybertruck/", language="text")
    with right:
        st.markdown("**Công cụ kiểm tra được gì?**")
        st.markdown("- Ảnh đọc được, kích thước và số ảnh từng lớp.\n- Nhãn thiếu/lạ, nguồn ảnh chưa khai báo.\n- Ảnh trùng byte hoặc pixel đã giải mã.\n- Ảnh trùng/cùng xe ở nhiều tập nếu có mã xe.\n- Báo cáo CSV và JSON để nhóm kiểm tra.")
    st.info("Không tự kết luận quyền sử dụng, độ chính xác nhãn hay khả năng nhận diện. Chưa có dataset được duyệt và chưa huấn luyện mô hình.")
    st.stop()

dataset_string, report = saved
summary = report["summary"]
st.caption(f"Dataset đã quét: {report['dataset_name']} · UTC: {report['scanned_at_utc']} · Quét lại sau khi thay đổi ảnh hoặc thư mục.")
if not summary["images"]:
    st.warning("Thư mục tồn tại nhưng chưa có ảnh hỗ trợ: JPG, PNG, WEBP, BMP hoặc TIFF. Chưa có dữ liệu để đánh giá.")
    st.stop()

columns = st.columns(4)
for column, label, value in zip(columns, ["Ảnh tìm thấy", "Ảnh đọc được", "Lớp có ảnh hợp lệ", "Ảnh lỗi đọc"],
                                [summary["images"], summary["readable"], summary["recognized_classes"], summary["issues"].get("unreadable", 0)]):
    column.metric(label, value)

overview, quality, gallery, exports = st.tabs(["Tổng quan", "Chất lượng", "Xem ảnh", "Báo cáo"])
with overview:
    st.subheader("Phân bố ảnh đọc được")
    frame = pd.DataFrame(report["classes"]).set_index("model")
    st.bar_chart(frame["readable_images"], color="#218aa7", horizontal=True)
    st.dataframe(frame.reset_index(), hide_index=True, width="stretch")
    st.caption("Đếm ảnh đọc được theo nhãn thư mục; ảnh trùng vẫn nằm trong số đếm. Không phải số ảnh độc lập đã duyệt.")
    if any(row["readable_images"] == 0 for row in report["classes"]):
        st.warning("Có lớp chưa có ảnh đọc được. Chưa đủ căn cứ chốt cả 5 lớp.")
    st.info("Biểu đồ cho biết phân bố, không chứng minh độ đa dạng góc chụp, ánh sáng hoặc độ đúng của nhãn.")

with quality:
    st.subheader("Các mục cần rà soát")
    issues = summary["issues"]
    labels = {"unreadable": "Ảnh lỗi đọc", "missing_label": "Ảnh không có thư mục lớp",
              "unknown_label": "Nhãn ngoài 5 lớp", "missing_provenance": "Thiếu nguồn hoặc giấy phép",
              "duplicate": "Ảnh thuộc nhóm trùng", "conflicting_labels": "Ảnh trùng nhưng nhãn khác",
              "split_leakage": "Ảnh/nhóm xe xuất hiện ở nhiều tập"}
    st.dataframe([{"Mục": name, "Số ảnh": issues.get(key, 0)} for key, name in labels.items()], hide_index=True, width="stretch")
    if summary["leakage_images"]:
        st.error("Phát hiện nhóm trùng/cùng xe xuất hiện ở nhiều tập. Cần xử lý trước khi đánh giá mô hình.")
    if summary["skipped_external_links"]:
        st.warning(f"Bỏ qua {summary['skipped_external_links']} liên kết ảnh trỏ ra ngoài thư mục đã chọn.")
    rows = [{**row, "issues": "; ".join(row["issues"])} for row in report["images"]]
    only_issues = st.checkbox("Chỉ hiện ảnh cần kiểm tra", value=True)
    st.dataframe([row for row in rows if row["issues"] or not only_issues], hide_index=True, width="stretch")
    with st.expander("Nhóm ảnh trùng và cảnh báo chia tập"):
        st.json({"duplicate_groups": report["duplicate_groups"], "leakage_groups": report["leakage_groups"]})
    st.caption("Một ảnh có thể có nhiều vấn đề. Không cộng các dòng thành tổng ảnh lỗi. Không xóa ảnh tự động.")

with gallery:
    st.subheader("Kiểm tra nhãn bằng mắt")
    selection = st.selectbox("Dòng xe", options=["Tất cả", *MODELS.values(), "Nhãn khác / thiếu"])
    candidates = [row for row in report["images"] if row["readable"] and
                  (selection == "Tất cả" or MODELS.get(row["class_id"], "Nhãn khác / thiếu") == selection)]
    if candidates:
        relative = st.selectbox("Ảnh", [row["relative_path"] for row in candidates])
        image_path = (Path(dataset_string) / relative).resolve()
        try:
            if not image_path.is_relative_to(Path(dataset_string)):
                raise ValueError("Ảnh không nằm trong thư mục đã quét.")
            with Image.open(image_path) as original:
                if original.width * original.height > 25_000_000:
                    raise ValueError("Ảnh quá lớn để xem trước.")
                preview = ImageOps.exif_transpose(original).convert("RGB")
                preview.thumbnail((1100, 700))
            st.image(preview, caption=relative)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            st.error(f"Không mở được ảnh. Quét lại nếu dữ liệu đã đổi: {type(error).__name__}")
        st.caption("Tên lớp là nhãn do dữ liệu cung cấp, không phải dự đoán AI. Công cụ chưa có chức năng sửa nhãn.")
    else:
        st.info("Chưa có ảnh đọc được cho lựa chọn này.")

with exports:
    st.subheader("Bằng chứng cho báo cáo Week 2")
    left, right = st.columns(2)
    left.download_button("Tải danh sách ảnh · CSV", report_csv(report), "images.csv", "text/csv", width="stretch")
    right.download_button("Tải báo cáo đầy đủ · JSON", json.dumps(report, ensure_ascii=False, indent=2), "audit.json", "application/json", width="stretch")
    st.markdown("**Nguồn ảnh tùy chọn:** tạo `provenance.csv` ngay trong thư mục dataset. Mỗi dòng ghi một ảnh; đường dẫn tính từ gốc dataset.")
    st.code("relative_path,source_url,license,vehicle_id\n", language="text")
    st.write("Khai báo nguồn không có nghĩa nguồn hoặc quyền sử dụng đã được xác minh. Không biết mã xe thì để trống, không gán chung một mã cho mọi xe cùng dòng.")
    st.warning("Chỉ phát hiện ảnh trùng byte/pixel. Ảnh đổi kích thước, crop hoặc chụp lại cùng xe cần rà soát thêm. Chưa có kiểm định nhãn, chia tập tự động hay huấn luyện mô hình.")
