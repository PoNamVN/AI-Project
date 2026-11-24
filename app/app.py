"""Tesla candidate recognition and dataset exploration; uploaded photos are not persisted."""

import json
import io
import hashlib
import os
import sys
import warnings
import threading
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_audit import MAX_PIXELS, MODELS, audit_dataset, report_csv
from src.cloud_data import CloudData


st.set_page_config(page_title="CarVision AI · Nhận diện Tesla", page_icon="🚘", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
st.markdown("""
<div class="cv-hero">
<div><div class="cv-brand"><svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true"><rect x="1" y="1" width="26" height="26" rx="8" stroke="#71d0d7"/><path d="M20 10a7 7 0 1 0 0 8M13 14h9" stroke="#9ae1e7" stroke-width="2" stroke-linecap="round"/></svg>CARVISION AI</div>
<h1>Chiếc Tesla này<br>là dòng xe nào?</h1>
<p>Thêm ảnh chiếc xe, xem các dòng gần giống và chọn mẫu bạn muốn tìm hiểu.</p>
<div class="cv-scope">Model S · Model 3 · Model X · Model Y · Cybertruck &nbsp; / &nbsp; Bản thử nghiệm</div></div>
<svg class="cv-hero-art" viewBox="0 0 360 180" fill="none" aria-hidden="true"><circle cx="230" cy="80" r="72" stroke="#234958"/><circle cx="230" cy="80" r="52" stroke="#234958"/><path d="M20 136h320M40 117l14-29 62-10 44-35h76l38 37 44 12 14 25v17h-26M100 134h147M40 134H28v-17h12M121 78h140M163 47l-24 30m55-30v30m39-30 26 30" stroke="#83d5dd" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="75" cy="131" r="24" fill="#102b3a" stroke="#c3edf0" stroke-width="2.5"/><circle cx="75" cy="131" r="10" stroke="#83d5dd" stroke-width="2"/><circle cx="274" cy="131" r="24" fill="#102b3a" stroke="#c3edf0" stroke-width="2.5"/><circle cx="274" cy="131" r="10" stroke="#83d5dd" stroke-width="2"/><path d="M300 100h14M49 99h15M200 89h13" stroke="#c3edf0" stroke-width="3" stroke-linecap="round"/></svg>
</div>
""", unsafe_allow_html=True)

source = "Nhận diện ảnh"
with st.sidebar:
    st.markdown("### CarVision AI")
    st.caption("Nhận diện thử nghiệm qua ảnh")
    if os.environ.get("CARVISION_ENABLE_DEV_TOOLS") == "1":
        source = st.radio("Chế độ dữ liệu", ["Trên máy", "Supabase", "Nhận diện ảnh"], index=2,
                          format_func=lambda value: {"Trên máy": "Rà dữ liệu trên máy", "Supabase": "Kho ảnh của nhóm", "Nhận diện ảnh": "Nhận diện xe"}[value])
        st.caption("Các mục rà dữ liệu phục vụ nhóm làm dự án; không cần dùng khi nhận diện xe.")
if source == "Nhận diện ảnh":
    from src.recognition_workflow import compare_views, reference_preview, reviewed_result

    run = Path(os.environ.get("CARVISION_VISION_RUN", str(ROOT / "data/processed/vision-recognition-v3-brand"))).resolve()
    if not run.is_relative_to((ROOT / "data/processed").resolve()) or not (run / "model.json").is_file():
        st.info("Ứng dụng chưa sẵn sàng nhận diện trên máy này. Cần người cài đặt bổ sung gói mô hình trước khi bắt đầu.")
        with st.expander("Hướng dẫn cho người cài đặt"):
            st.write("Chuẩn bị môi trường và gói runtime theo docs/team_setup.md, sau đó chạy npm run check.")
        st.stop()

    @st.cache_resource(show_spinner="Đang nạp mô hình nhận diện trên máy chủ…")
    def load_recognizer(model_path, version):
        from src.recognize_vehicle import Recognizer
        return Recognizer(Path(model_path))

    @st.cache_resource
    def inference_slot():
        # Keep shared CPU inference from multiplying activation memory across visitors.
        return threading.BoundedSemaphore(1)

    @st.cache_data(ttl=900, max_entries=32, show_spinner=False)
    def cloud_preview(reference, credential_epoch):
        return reference_preview(reference, ROOT / "data/raw", True)

    reference_source = "Supabase riêng tư" if os.environ.get("CARVISION_SUPABASE_URL") and os.environ.get("CARVISION_SUPABASE_KEY") else "Trên máy chủ"
    credential_epoch = hashlib.sha256((os.environ.get("CARVISION_SUPABASE_URL", "") + "\n" + os.environ.get("CARVISION_SUPABASE_KEY", "")).encode()).hexdigest()
    upload_column, help_column = st.columns([1.35, .85], gap="large")
    with upload_column:
        photo_panel = st.container(border=True, key="photo-panel")
        with photo_panel:
            st.markdown('<div class="cv-step">BƯỚC 1</div>', unsafe_allow_html=True)
            st.subheader("Thêm ảnh chiếc xe", anchor=False)
            st.caption("Ưu tiên ảnh rõ một chiếc xe, trọn thân và đủ sáng.")
            input_source = st.radio("Ảnh chiếc xe", ["Chọn từ album / máy", "Chụp trực tiếp"], horizontal=True, label_visibility="collapsed")
            if input_source == "Chụp trực tiếp":
                st.caption("Cho phép trình duyệt dùng camera, rồi chụp trọn chiếc xe. Nếu camera chưa dùng được, chọn ảnh từ album.")
                uploaded = st.camera_input("Chụp trọn một chiếc xe")
            else:
                uploaded = st.file_uploader("Chọn ảnh chiếc xe", type=["jpg", "jpeg", "png", "webp"], max_upload_size=30,
                                            help="JPG, PNG hoặc WEBP, tối đa 30 MB. Ảnh iPhone dạng HEIC cần xuất sang JPG trước.")
            st.caption("Ảnh được xử lý trong phiên này; không tự lưu vào kho ảnh hoặc dùng để AI học lại.")
            with st.expander("Tùy chọn phân tích"):
                detailed = st.checkbox("Giải thích chi tiết và bản đồ vùng ảnh ảnh hưởng", value=False)
                st.caption("Mặc định phân tích nhanh. Bật mục này rồi phân tích lại để xem bản đồ ảnh; xử lý sẽ lâu hơn.")
    with help_column, st.container(border=True, key="howto-panel"):
        st.markdown('''<div class="cv-help-title">Một ảnh rõ, dễ đối chiếu hơn</div>
<div class="cv-help-item"><span class="cv-help-number">1</span><div><strong>Chụp trọn chiếc xe</strong>Ưu tiên một xe, đủ sáng, góc chéo trước hoặc sau.</div></div>
<div class="cv-help-item"><span class="cv-help-number">2</span><div><strong>Xem các mẫu gần giống</strong>Đối chiếu dáng xe và ảnh tham khảo trước khi chọn.</div></div>
<div class="cv-help-item"><span class="cv-help-number">3</span><div><strong>Chưa chắc? Thêm góc khác</strong>Bạn có thể chọn chưa xác định hoặc bổ sung một ảnh nữa.</div></div>
<div class="cv-help-foot">Hiện hỗ trợ 5 dòng Tesla. Ảnh chưa đủ để xác định chính xác đời xe, phiên bản hoặc thông số.</div>''', unsafe_allow_html=True)
    if uploaded is None:
        for key in ("recognition", "vehicle_review", "reference_previews", "recognition_context", "second_view"):
            st.session_state.pop(key, None)
        with photo_panel:
            st.button("Phân tích ảnh", type="primary", disabled=True, width="stretch")
        st.markdown('<div class="cv-empty"><strong>Kết quả sẽ hiện ở đây</strong><p>Chọn ảnh ở trên, sau đó bấm Phân tích ảnh để xem các dòng xe gợi ý.</p></div>', unsafe_allow_html=True)
        st.stop()
    data = uploaded.getvalue()
    try:
        from src.vision_features import open_image
        preview = open_image(data)
        preview.thumbnail((760, 440))
        with photo_panel:
            st.image(preview, caption="Ảnh bạn đã chọn", width="stretch")
    except (ValueError, OSError) as error:
        for key in ("recognition", "vehicle_review", "reference_previews", "recognition_context", "second_view"):
            st.session_state.pop(key, None)
        with photo_panel:
            st.error("Chưa mở được ảnh này. Hãy thử ảnh JPG, PNG hoặc WEBP khác, tối đa 30 MB và 25 triệu điểm ảnh.")
            with st.expander("Chi tiết lỗi ảnh"):
                st.caption(str(error))
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
        selected_index = photo_panel.selectbox("Chọn xe theo số thứ tự khung", range(len(saved[1]["detections"])), format_func=lambda index: f"Xe {index+1}")
        from PIL import ImageDraw
        boxes = open_image(data)
        drawing = ImageDraw.Draw(boxes)
        for index, detection in enumerate(saved[1]["detections"]):
            drawing.rectangle(detection["box"], outline="#218aa7", width=max(3, boxes.width//250))
            drawing.text(detection["box"][:2], f"{index+1}", fill="red", stroke_width=2)
        boxes.thumbnail((1100, 700))
        photo_panel.image(boxes, caption="Chọn chiếc xe muốn tìm hiểu")
    if photo_panel.button("Phân tích xe đã chọn" if selected_index is not None else "Phân tích ảnh", type="primary", width="stretch"):
        for key in ("recognition", "vehicle_review", "reference_previews", "second_view"):
            st.session_state.pop(key, None)
        try:
            with st.spinner("Đang xem chiếc xe trong ảnh… Lần đầu có thể cần thêm thời gian nạp mô hình."):
                version = (run / "model.json").stat().st_mtime_ns
                recognizer = load_recognizer(str(run), version)
                with inference_slot():
                    result, annotated, heatmap = recognizer.analyze(data, selected_index, detailed)
                metadata = getattr(recognizer, "metadata", {})
                result = {**result, "model_manifest_sha256": metadata.get("manifest_sha256") if isinstance(metadata, dict) else None}
                st.session_state.recognition = (image_id, result, annotated, heatmap)
            if result["status"] == "multiple_vehicles" and selected_index is None:
                st.rerun()
        except ImportError:
            st.error("Máy chủ chưa đủ thành phần nhận diện. Người cài đặt xem hướng dẫn trong docs/team_setup.md.")
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Chưa phân tích được ảnh: {error}")
    saved = st.session_state.get("recognition")
    if saved:
        _, result, annotated, heatmap = saved
        with st.container(border=True, key="result-summary"):
            st.markdown('<div class="cv-step">BƯỚC 2 · KẾT QUẢ GỢI Ý</div>', unsafe_allow_html=True)
            if result.get("candidates"):
                st.subheader("Gần giống Tesla " + result["candidates"][0]["model"], anchor=False)
                st.caption("Đây là gợi ý để đối chiếu, chưa phải xác nhận dòng xe. Các mẫu gần giống nhau vẫn có thể bị nhận nhầm.")
            else:
                st.subheader("Cần xem rõ chiếc xe hơn", anchor=False)
            if result["status"] == "tesla_candidate":
                st.info(result["message"])
            else:
                st.warning(result["message"])
        if result.get("candidates"):
            st.markdown('<div class="cv-step">BƯỚC 3</div>', unsafe_allow_html=True)
            st.subheader("Bạn thấy mẫu nào giống nhất?", anchor=False)
            st.caption("Chọn mẫu để ghi nhận ý kiến của bạn. Ảnh tham khảo lấy từ bộ ảnh AI đã học; không xác nhận đời xe, phiên bản hoặc cùng một chiếc xe.")
            references = {r["class_id"]: r for r in result.get("candidate_references", [])}
            previews = st.session_state.setdefault("reference_previews", {})
            for index, (column, candidate) in enumerate(zip(st.columns(3), result["candidates"]), 1):
                with column, st.container(border=True, key=f"candidate-{index}"):
                    st.markdown('<span class="cv-badge">' + ("Gợi ý đầu tiên" if index == 1 else f"Gợi ý {index}") + '</span>', unsafe_allow_html=True)
                    st.markdown(f"### Tesla {candidate['model']}")
                    reference = references.get(candidate.get("class_id"))
                    if reference:
                        cache_key = (reference_source, credential_epoch, reference["source_path"], reference.get("sha256", ""))
                        if cache_key not in previews:
                            if len(previews) >= 3:
                                previews.clear()
                            try:
                                previews[cache_key] = cloud_preview(reference, credential_epoch) if reference_source == "Supabase riêng tư" else reference_preview(reference, ROOT / "data/raw", False)
                            except (ValueError, OSError) as error:
                                previews[cache_key] = str(error)
                        photo = previews[cache_key]
                        if isinstance(photo, str):
                            st.warning(photo)
                        else:
                            st.image(photo, caption=f"Ảnh tham chiếu · {reference['model']}")
                        st.caption("Ảnh tham khảo · " + reference["license"])
                        st.link_button("Nguồn ảnh", reference["source_url"])
                        with st.expander("Ghi công ảnh"):
                            st.json(reference)
                    else:
                        st.info("Chưa tải được ảnh tham khảo cho dòng này. Bạn vẫn có thể đọc gợi ý hoặc thử ảnh khác.")
                    if candidate.get("class_id") in MODELS and st.button("Chọn " + candidate["model"], key="choose_" + candidate["class_id"],
                                                                        type="primary" if index == 1 else "secondary", width="stretch"):
                        st.session_state.vehicle_review = reviewed_result(result, candidate["class_id"])
            if st.button("Không có mẫu phù hợp", width="stretch"):
                st.session_state.vehicle_review = reviewed_result(result, None)
            with st.expander("Chọn dòng khác hoặc tải lại ảnh tham khảo"):
                choices = [None, *MODELS]
                choice = st.selectbox("Theo bạn, đây là dòng nào?", choices,
                                      format_func=lambda value: "Không có mẫu phù hợp / chưa xác định" if value is None else "Tesla " + MODELS[value],
                                      key="review_choice_" + image_id)
                if st.button("Lưu lựa chọn của tôi"):
                    st.session_state.vehicle_review = reviewed_result(result, choice)
                if st.button("Tải lại ảnh đối chiếu"):
                    st.session_state.pop("reference_previews", None)
                    st.rerun()
            reviewed = st.session_state.get("vehicle_review")
            if reviewed:
                selected = reviewed["user_review"]["model"]
                st.success("Đã ghi nhận trong phiên này: " + ("Tesla " + selected if selected else "Chưa xác định được mẫu phù hợp"))
            st.caption("Ý kiến của bạn không thay đổi dự đoán AI và chưa được xác minh. Tải kết quả bên dưới nếu muốn giữ lại khi đóng trang.")
            with st.expander("Vì sao chưa có đời xe và thông số?"):
                st.write("Một bức ảnh thường không đủ để biết chính xác năm, phiên bản, pin hoặc động cơ. Cần thông tin về xe và tài liệu hãng; công cụ hiện chưa có phần tra cứu này.")
            with st.expander("Vẫn phân vân? Thêm một góc ảnh"):
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
                                    with inference_slot():
                                        extra, _, _ = load_recognizer(str(run), context[3]).analyze(second_data, None, False)
                                st.session_state.second_view = {"input_sha256": second_id, "result": extra, "comparison": compare_views(result, extra), "same_vehicle": "user_declared_not_verified"}
                            except (ImportError, ValueError, OSError, KeyError) as error:
                                st.error(f"Chưa đối chiếu được góc thứ hai: {error}")
                extra = st.session_state.get("second_view") if same_vehicle else None
                if extra:
                    st.info(extra["comparison"]["message"])
                    st.caption("Ảnh thứ hai: " + extra["result"]["message"])
        exported = st.session_state.get("vehicle_review", result)
        exported = {**exported, "input_sha256": image_id}
        if result.get("candidates") and same_vehicle and st.session_state.get("second_view"):
            exported["second_view"] = st.session_state.second_view
        with st.expander("Xem cách AI đưa ra gợi ý"):
            if result.get("candidates"):
                st.dataframe([{"Dòng xe gợi ý": row["model"], "Điểm xếp hạng": round(row["ranking_score"], 4)} for row in result["candidates"]], hide_index=True, width="stretch")
                st.caption(result["note"])
                st.caption("Điểm dùng sắp xếp các gợi ý, không phải phần trăm chắc chắn đoán đúng.")
                for text in result["explanation"]:
                    st.write(text)
                if heatmap:
                    st.image(heatmap, caption="Đỏ: che vùng này làm điểm ứng viên giảm. Không phải bản đồ xác nhận bộ phận xe.")
                elif not detailed:
                    st.info("Muốn xem bản đồ vùng ảnh: mở Tùy chọn phân tích, bật giải thích chi tiết rồi bấm phân tích lại.")
                st.caption("Các mô tả sau là giả thuyết ảnh–văn bản, chưa xác nhận sự hiện diện của bộ phận.")
                st.dataframe([{"Nhóm": p["category"], "Mô tả gợi ý": p["hypothesis"]} for p in result["visual_hypotheses"]], hide_index=True, width="stretch")
                st.markdown("**Các ảnh học gần giống nhất**")
                st.caption("Có thể cùng một dòng; đây là bằng chứng tương đồng, không xác nhận đời xe hay phiên bản.")
                st.json(result["references"])
            if annotated:
                annotated = annotated.copy()
                annotated.thumbnail((1100, 700))
                st.image(annotated, caption="Vùng xe được chọn để phân tích")
            st.markdown("**Dữ liệu kết quả và giới hạn**")
            st.json(exported)
        st.download_button("Tải kết quả · JSON", json.dumps(exported, ensure_ascii=False, indent=2), "carvision-result.json", "application/json", width="stretch")
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
