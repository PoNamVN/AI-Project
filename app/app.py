"""Tesla candidate recognition and dataset exploration; uploaded photos are not persisted."""

import json
import io
import hashlib
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_audit import MAX_PIXELS, MODELS, audit_dataset, report_csv
from src.cloud_data import CloudData


st.set_page_config(page_title="CarVision AI · Tesla", page_icon="🚘", layout="wide")
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
<div class="cv-kicker">CARVISION AI / TESLA</div>
<h1>CarVision AI</h1>
<p>Nhận diện thử nghiệm qua ảnh, đối chiếu bằng chứng và kiểm tra dữ liệu Tesla.</p>
</div>
""", unsafe_allow_html=True)

source = st.sidebar.radio("Chế độ dữ liệu", ["Trên máy", "Supabase", "Nhận diện ảnh"], index=2)
if source == "Nhận diện ảnh":
    from src.recognition_workflow import compare_views, reference_preview, reviewed_result

    st.subheader("Nhận diện Tesla · đối chiếu bằng ảnh")
    st.caption("Bản thử nghiệm 5 dòng Tesla. Ảnh được xử lý tại máy chủ đang chạy app; không tự lưu ảnh tải lên vào dataset hoặc Supabase.")
    run = Path(os.environ.get("CARVISION_VISION_RUN", str(ROOT / "data/processed/vision-recognition-v3-brand"))).resolve()
    if not run.is_relative_to((ROOT / "data/processed").resolve()) or not (run / "model.json").is_file():
        st.info("Chưa có model nhận diện V3 trên máy này. Làm theo docs/vision_recognition.md để chuẩn bị model.")
        st.stop()

    @st.cache_resource(show_spinner="Đang nạp mô hình nhận diện trên máy chủ…")
    def load_recognizer(model_path, version):
        from src.recognize_vehicle import Recognizer
        return Recognizer(Path(model_path))

    input_source = st.radio("Ảnh chiếc xe", ["Chọn từ album / máy", "Chụp trực tiếp"], horizontal=True)
    if input_source == "Chụp trực tiếp":
        st.caption("Cho phép trình duyệt dùng camera. Điện thoại truy cập qua mạng cần HTTPS; localhost có thể dùng camera trên máy.")
        uploaded = st.camera_input("Chụp trọn một chiếc xe")
    else:
        uploaded = st.file_uploader("Chọn ảnh chiếc xe", type=["jpg", "jpeg", "png", "webp"], max_upload_size=30)
    detailed = st.checkbox("Giải thích chi tiết và bản đồ vùng ảnh ảnh hưởng", value=True)
    reference_source = st.radio("Nguồn ảnh đối chiếu", ["Trên máy chủ", "Supabase riêng tư"], horizontal=True)
    st.caption("Chỉ hiển thị tối đa 3 ảnh tham chiếu; người xem không tải toàn bộ kho ảnh hoặc model.")
    if uploaded is None:
        for key in ("recognition", "vehicle_review", "reference_previews", "recognition_context", "second_view"):
            st.session_state.pop(key, None)
        st.info("Chụp rõ một chiếc xe, ưu tiên góc chéo trước hoặc chéo sau và trọn thân xe.")
        st.stop()
    data = uploaded.getvalue()
    try:
        from src.vision_features import open_image
        preview = open_image(data)
        preview.thumbnail((1100, 700))
        st.image(preview, caption="Ảnh bạn cung cấp")
    except (ValueError, OSError) as error:
        st.error(str(error))
        st.stop()
    image_id = hashlib.sha256(data).hexdigest()
    context = (image_id, input_source, detailed, (run / "model.json").stat().st_mtime_ns)
    saved = st.session_state.get("recognition")
    if st.session_state.get("recognition_context") != context:
        for key in ("recognition", "vehicle_review", "reference_previews", "second_view"):
            st.session_state.pop(key, None)
        saved = None
    st.session_state.recognition_context = context
    selected_index = None
    if saved and saved[1]["status"] == "multiple_vehicles":
        selected_index = st.selectbox("Chọn xe theo số thứ tự khung", range(len(saved[1]["detections"])), format_func=lambda index: f"Xe {index+1}")
        from PIL import ImageDraw
        boxes = open_image(data)
        drawing = ImageDraw.Draw(boxes)
        for index, detection in enumerate(saved[1]["detections"]):
            drawing.rectangle(detection["box"], outline="#218aa7", width=max(3, boxes.width//250))
            drawing.text(detection["box"][:2], f"{index+1}", fill="red", stroke_width=2)
        boxes.thumbnail((1100, 700))
        st.image(boxes, caption="Chọn chiếc xe muốn tìm hiểu")
    if st.button("Phân tích xe đã chọn" if selected_index is not None else "Phân tích ảnh", type="primary"):
        for key in ("recognition", "vehicle_review", "reference_previews", "second_view"):
            st.session_state.pop(key, None)
        try:
            with st.spinner("Đang phân tích chiếc xe và kiểm tra bằng chứng hình ảnh…"):
                version = (run / "model.json").stat().st_mtime_ns
                recognizer = load_recognizer(str(run), version)
                result, annotated, heatmap = recognizer.analyze(data, selected_index, detailed)
                metadata = getattr(recognizer, "metadata", {})
                result = {**result, "model_manifest_sha256": metadata.get("manifest_sha256") if isinstance(metadata, dict) else None}
                st.session_state.recognition = (image_id, result, annotated, heatmap)
            if result["status"] == "multiple_vehicles" and selected_index is None:
                st.rerun()
        except ImportError:
            st.error("Môi trường hiện tại chưa có thư viện vision. Cài requirements-demo.txt và requirements-vision.txt theo hướng dẫn.")
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Chưa phân tích được ảnh: {error}")
    saved = st.session_state.get("recognition")
    if saved:
        _, result, annotated, heatmap = saved
        if result["status"] == "tesla_candidate":
            st.success(result["message"])
        else:
            st.warning(result["message"])
        if result.get("candidates"):
            st.write("**Gợi ý để bạn đối chiếu:** Tesla " + result["candidates"][0]["model"])
            st.caption("Có thể nhận nhầm các xe giống nhau. Hãy đối chiếu ảnh mẫu; nếu chưa chắc, thêm góc chụp phía trước hoặc phía sau của cùng chiếc xe.")
        if annotated:
            annotated = annotated.copy()
            annotated.thumbnail((1100, 700))
            st.image(annotated, caption="Vùng xe được chọn để phân tích")
        if result.get("candidates"):
            st.dataframe([{"Ứng viên": row["model"], "Điểm xếp hạng": round(row["ranking_score"], 4)} for row in result["candidates"]], hide_index=True, width="stretch")
            st.caption(result["note"])
            st.subheader("Đối chiếu trước khi xác nhận")
            st.caption("Mỗi ứng viên có một ảnh train cùng dòng để tham khảo. Ảnh mẫu không xác nhận đời xe, phiên bản hoặc cùng một chiếc xe.")
            references = {r["class_id"]: r for r in result.get("candidate_references", [])}
            previews = st.session_state.setdefault("reference_previews", {})
            for index, (column, candidate) in enumerate(zip(st.columns(3), result["candidates"]), 1):
                with column:
                    st.markdown(f"**{index}. Tesla {candidate['model']}**")
                    reference = references.get(candidate.get("class_id"))
                    if reference:
                        cache_key = (reference_source, reference["source_path"], reference.get("sha256", ""))
                        if cache_key not in previews:
                            if len(previews) >= 3:
                                previews.clear()
                            try:
                                previews[cache_key] = reference_preview(reference, ROOT / "data/raw", reference_source == "Supabase riêng tư")
                            except (ValueError, OSError) as error:
                                previews[cache_key] = str(error)
                        photo = previews[cache_key]
                        if isinstance(photo, str):
                            st.warning(photo)
                        else:
                            st.image(photo, caption=f"Ảnh tham chiếu · {reference['model']}")
                        st.caption(reference["license"])
                        st.link_button("Nguồn ảnh", reference["source_url"])
                        with st.expander("Ghi công ảnh"):
                            st.json(reference)
                    else:
                        st.info("Chưa có ảnh tham chiếu phù hợp trong artifact này.")
            if st.button("Tải lại ảnh đối chiếu"):
                st.session_state.pop("reference_previews", None)
                st.rerun()
            choices = [None, *MODELS]
            choice = st.selectbox("Theo bạn, đây là dòng nào?", choices,
                                  format_func=lambda value: "Không có mẫu phù hợp / chưa xác định" if value is None else "Tesla " + MODELS[value],
                                  key="review_choice_" + image_id)
            if st.button("Lưu lựa chọn của tôi", type="primary"):
                st.session_state.vehicle_review = reviewed_result(result, choice)
            reviewed = st.session_state.get("vehicle_review")
            if reviewed:
                selected = reviewed["user_review"]["model"]
                st.info("Bạn chọn: " + ("Tesla " + selected if selected else "Không có mẫu phù hợp / chưa xác định"))
            st.caption("Lựa chọn chỉ nằm trong phiên hiện tại và bản JSON bạn tải; không tự sửa dự đoán, không lưu ảnh hay dùng để huấn luyện.")
            st.caption("Đời xe, phiên bản, pin và thông số: chưa xác định. Cần năm/thị trường/phiên bản và tài liệu hãng trước khi tra cứu.")
            st.subheader("Thêm một góc ảnh để đối chiếu")
            st.caption("Dùng ảnh khác của cùng chiếc xe. Hệ thống so sánh hai kết quả độc lập, không cộng điểm và không tự tăng độ tin cậy.")
            second = st.file_uploader("Ảnh góc thứ hai", type=["jpg", "jpeg", "png", "webp"], max_upload_size=30, key="second_photo")
            same_vehicle = st.checkbox("Tôi xác nhận hai ảnh chụp cùng một chiếc xe", key="same_vehicle_" + image_id)
            if second is None:
                st.session_state.pop("second_view", None)
            else:
                second_data = second.getvalue()
                second_id = hashlib.sha256(second_data).hexdigest()
                existing = st.session_state.get("second_view")
                if existing and existing["input_sha256"] != second_id:
                    st.session_state.pop("second_view", None)
                if st.button("Đối chiếu hai góc ảnh"):
                    st.session_state.pop("second_view", None)
                    if not same_vehicle:
                        st.warning("Chỉ đối chiếu khi bạn biết hai ảnh là cùng một chiếc xe.")
                    elif second_id == image_id:
                        st.warning("Đây là cùng một tệp ảnh. Hãy chọn góc chụp khác.")
                    else:
                        try:
                            with st.spinner("Đang phân tích góc ảnh thứ hai…"):
                                extra, _, _ = load_recognizer(str(run), context[3]).analyze(second_data, None, False)
                            st.session_state.second_view = {"input_sha256": second_id, "result": extra, "comparison": compare_views(result, extra), "same_vehicle": "user_declared_not_verified"}
                        except (ImportError, ValueError, OSError, KeyError) as error:
                            st.error(f"Chưa đối chiếu được góc thứ hai: {error}")
            extra = st.session_state.get("second_view") if same_vehicle else None
            if extra:
                st.info(extra["comparison"]["message"])
                st.caption("Ảnh thứ hai: " + extra["result"]["message"])
            with st.expander("Giải thích kỹ thuật và bản đồ vùng ảnh ảnh hưởng"):
                for text in result["explanation"]:
                    st.write(text)
                if heatmap:
                    st.image(heatmap, caption="Đỏ: che vùng này làm điểm ứng viên giảm. Không phải bản đồ xác nhận bộ phận xe.")
                st.caption("Các mô tả sau là giả thuyết ảnh–văn bản, chưa xác nhận sự hiện diện của bộ phận.")
                st.dataframe([{"Nhóm": p["category"], "Mô tả gợi ý": p["hypothesis"]} for p in result["visual_hypotheses"]], hide_index=True, width="stretch")
            with st.expander("Ba ảnh train gần nhất toàn bộ · bằng chứng tương đồng"):
                st.caption("Có thể cùng một dòng, khác với ba ảnh đại diện cho ứng viên ở phía trên.")
                st.json(result["references"])
        exported = st.session_state.get("vehicle_review", result)
        exported = {**exported, "input_sha256": image_id}
        if result.get("candidates") and same_vehicle and st.session_state.get("second_view"):
            exported["second_view"] = st.session_state.second_view
        with st.expander("Chi tiết kết quả và giới hạn"):
            st.json(exported)
        st.download_button("Tải giải thích · JSON", json.dumps(exported, ensure_ascii=False, indent=2), "carvision-result.json", "application/json")
    st.stop()

if source == "Supabase":
    st.subheader("Kho ảnh Tesla trên Supabase")
    st.caption("Chỉ tải danh mục và ảnh đang chọn. Nhãn là ứng viên đã rà ban đầu, chưa phải dữ liệu được duyệt để huấn luyện.")
    if st.button("Nạp / cập nhật danh mục", type="primary"):
        st.session_state.pop("cloud_catalog", None)
        st.session_state.pop("cloud_preview", None)
        try:
            with st.spinner("Đang đọc danh mục…"):
                st.session_state.cloud_catalog = CloudData().catalog()
        except ValueError as error:
            st.error(str(error))
    rows = st.session_state.get("cloud_catalog", [])
    if not rows:
        st.info("Bấm nạp danh mục sau khi cấu hình kết nối Supabase trên máy chủ. Hướng dẫn: docs/supabase_data.md.")
        st.stop()
    dataset = st.selectbox("Phiên bản dữ liệu", sorted({row["dataset"] for row in rows}))
    label = st.selectbox("Dòng xe trên cloud", ["Tất cả", *MODELS.values()])
    candidates = [row for row in rows if row["dataset"] == dataset and row["decision"] == "keep"
                  and (label == "Tất cả" or MODELS.get(row["class_id"]) == label)]
    if candidates:
        relative = st.selectbox("Ảnh trên cloud", [row["relative_path"] for row in candidates])
        row = next(row for row in candidates if row["relative_path"] == relative)
        try:
            saved_preview = st.session_state.get("cloud_preview")
            if saved_preview is None or saved_preview[0] != row["sha256"]:
                data = CloudData().image(row)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(data)) as original:
                        if original.width * original.height > MAX_PIXELS:
                            raise ValueError("Ảnh vượt giới hạn xem trước.")
                        preview = ImageOps.exif_transpose(original).convert("RGB")
                        preview.thumbnail((1100, 700))
                st.session_state.cloud_preview = (row["sha256"], preview)
            st.image(st.session_state.cloud_preview[1], caption=relative)
            st.caption(f"Mục đích: {row['purpose']} · Tập cũ: {row['split']} · Giấy phép: {row['license']}")
            with st.expander("Nguồn, tác giả và trạng thái rà soát"):
                st.json(row["provenance"])
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            st.error(str(error))
    else:
        st.info("Chưa có ảnh đề xuất giữ cho lựa chọn này.")
    st.warning("Bộ evaluation_only dùng riêng để đánh giá. Ảnh supplement_pending cần xác nhận nhãn và nhóm xe trước khi học.")
    st.stop()

with st.sidebar:
    st.subheader("Nguồn dữ liệu")
    folder = st.text_input("Thư mục bên trong data/raw", value="tesla", help="Ví dụ tesla hoặc một thư mục con đã có ảnh.")
    scan = st.button("Quét dữ liệu", type="primary", width="stretch")
    st.caption("Chỉ đọc ảnh trên máy. Không tải ảnh lên dịch vụ ngoài và không sửa ảnh gốc.")
    st.divider()
    st.markdown("**5 lớp ứng viên**")
    st.write("Model S · Model 3 · Model X · Model Y · Cybertruck")
    st.caption("Nhãn thư mục cần được người hiểu xe kiểm tra. Chế độ này chỉ kiểm tra dữ liệu; chọn Nhận diện ảnh để thử model.")

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
    st.info("Không tự kết luận quyền sử dụng hoặc độ đúng của nhãn. Dataset chưa được duyệt độc lập; các model hiện có vẫn là thử nghiệm.")
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
    st.warning("Chỉ phát hiện ảnh trùng byte/pixel. Ảnh đổi kích thước, crop hoặc chụp lại cùng xe cần rà soát thêm. Chế độ audit không tự duyệt nhãn, chia tập hay huấn luyện model.")
