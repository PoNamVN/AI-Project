"""Demo workflow: bounded reference images and explicit, unverified user choices."""

import hashlib
from pathlib import Path, PurePosixPath

from src.cloud_data import CloudData
from src.data_audit import MODELS
from src.vision_features import open_image


def reference_preview(reference, raw_root, use_cloud=False):
    source = reference.get("source_path", "")
    if not isinstance(source, str):
        raise ValueError("Đường dẫn ảnh tham chiếu không hợp lệ.")
    parts = PurePosixPath(source).parts
    if (len(parts) != 3 or ".." in parts or "\\" in source or ":" in source
            or reference.get("class_id") not in MODELS or parts[1] != reference["class_id"]):
        raise ValueError("Đường dẫn ảnh tham chiếu không hợp lệ.")
    if use_cloud:
        data = CloudData().reference_image(reference.get("sha256", ""), reference["class_id"], thumbnail=True)
    else:
        root = Path(raw_root).resolve()
        path = (root / source).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Ảnh tham chiếu nằm ngoài kho dữ liệu.")
        with path.open("rb") as handle:
            data = handle.read(30_000_001)
        if len(data) > 30_000_000:
            raise ValueError("Ảnh tham chiếu vượt giới hạn 30 MB.")
        expected = reference.get("sha256")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Ảnh tham chiếu local không khớp checksum của mô hình.")
    image = open_image(data)
    image.thumbnail((500, 320))
    return image


def reviewed_result(result, class_id):
    """Keep model output intact; a user choice never becomes a training label."""
    if not result.get("candidates") or (class_id is not None and (not isinstance(class_id, str) or class_id not in MODELS)):
        raise ValueError("Kết quả hoặc dòng xe xác nhận không hợp lệ.")
    return {**result, "user_review": {
        "decision": "no_match" if class_id is None else "selected",
        "class_id": class_id,
        "model": MODELS.get(class_id),
        "verification": "user_choice_not_verified",
        "specifications": None,
    }}


def compare_views(first, second):
    """Compare independent suggestions; never fuse uncalibrated scores."""
    allowed = {"tesla_candidate", "uncertain_model"}
    if (first.get("status") not in allowed or second.get("status") not in allowed
            or not first.get("candidates") or not second.get("candidates")):
        return {"status": "insufficient", "message": "Chưa đủ căn cứ để đối chiếu hai ảnh. Hãy chụp rõ một xe trong mỗi ảnh; kiểm tra thông báo của ảnh góc thứ hai."}
    a, b = first["candidates"][0].get("class_id"), second["candidates"][0].get("class_id")
    if not isinstance(a, str) or not isinstance(b, str) or a not in MODELS or b not in MODELS:
        raise ValueError("Nhãn kết quả hai góc ảnh không hợp lệ.")
    if a != b:
        status = "disagreement"
        message = f"Hai ảnh chưa thống nhất: ảnh đầu gợi ý {MODELS[a]}, ảnh thứ hai gợi ý {MODELS[b]}. Cần đối chiếu thêm, chưa nên chốt dòng xe."
    elif first["status"] != "tesla_candidate" or second["status"] != "tesla_candidate":
        status = "same_but_uncertain"
        message = f"Hai ảnh cùng gợi ý {MODELS[a]}, nhưng ít nhất một ảnh còn chưa chắc chắn. Chưa xác nhận dòng xe."
    else:
        status = "agreement"
        message = f"Hai góc ảnh cùng gợi ý {MODELS[a]}. Đây là bằng chứng đối chiếu thêm, không phải xác nhận chính xác hoặc tăng phần trăm tin cậy."
    return {"status": status, "first_model": MODELS[a], "second_model": MODELS[b], "message": message}
