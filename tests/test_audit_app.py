"""UI checks use synthetic images only, never project dataset evidence."""
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import hashlib
import io
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_week5_album_camera_confirmation_and_reference_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/processed") as directory:
            run = Path(directory)
            (run / "model.json").write_text("{}")
            photo = Image.new("RGB", (40, 30), "blue")
            uploaded = io.BytesIO()
            photo.save(uploaded, format="PNG")
            other = io.BytesIO()
            Image.new("RGB", (40, 30), "red").save(other, format="PNG")
            labels = [("model_3", "Model 3"), ("model_y", "Model Y"), ("model_s", "Model S")]
            result = dict(status="tesla_candidate", message="Ứng viên Tesla", note="Chưa hiệu chuẩn",
                          candidates=[dict(class_id=c, model=m, ranking_score=.8 / (i + 1)) for i, (c, m) in enumerate(labels)],
                          references=[], explanation=[], visual_hypotheses=[],
                          candidate_references=[dict(class_id=c, model=m, source_path=f"fixture/{c}/car.png",
                                                     sha256="a" * 64, license="CC0", source_url="https://example.org/test") for c, m in labels])
            engine = Mock(metadata={"manifest_sha256": "b" * 64})
            engine.analyze.return_value = (result, None, None)
            with patch.dict("os.environ", {"CARVISION_VISION_RUN": str(run)}), \
                    patch("streamlit.file_uploader", return_value=uploaded) as uploader, \
                    patch("streamlit.camera_input", return_value=uploaded) as camera, \
                    patch("src.recognize_vehicle.Recognizer", return_value=engine), \
                    patch("src.recognition_workflow.reference_preview", return_value=photo) as preview:
                app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
                app.sidebar.radio[0].set_value("Nhận diện ảnh").run()
                next(b for b in app.button if b.label == "Phân tích ảnh").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(preview.call_count, 3)
                app.run()
                self.assertEqual(preview.call_count, 3)
                next(c for c in app.checkbox if c.label == "Tôi xác nhận hai ảnh chụp cùng một chiếc xe").set_value(True).run()
                next(b for b in app.button if b.label == "Đối chiếu hai góc ảnh").click().run()
                self.assertNotIn("second_view", app.session_state)
                self.assertTrue(any("cùng một tệp" in w.value for w in app.warning))
                uploader.side_effect = lambda label, **kwargs: other if label == "Ảnh góc thứ hai" else uploaded
                app.run()
                next(b for b in app.button if b.label == "Đối chiếu hai góc ảnh").click().run()
                self.assertEqual(app.session_state["second_view"]["comparison"]["status"], "agreement")
                self.assertEqual(engine.analyze.call_args.args[2], False)
                self.assertEqual(app.session_state["recognition"][1]["candidates"][0]["class_id"], "model_3")
                uploader.side_effect = None
                next(s for s in app.selectbox if s.label == "Theo bạn, đây là dòng nào?").set_value("model_y").run()
                next(b for b in app.button if b.label == "Lưu lựa chọn của tôi").click().run()
                reviewed = app.session_state["vehicle_review"]
                self.assertEqual(reviewed["user_review"]["class_id"], "model_y")
                self.assertEqual(reviewed["candidates"][0]["class_id"], "model_3")
                next(s for s in app.selectbox if s.label == "Theo bạn, đây là dòng nào?").set_value(None).run()
                next(b for b in app.button if b.label == "Lưu lựa chọn của tôi").click().run()
                self.assertEqual(app.session_state["vehicle_review"]["user_review"]["decision"], "no_match")
                next(r for r in app.radio if r.label == "Nguồn ảnh đối chiếu").set_value("Supabase riêng tư").run()
                self.assertEqual(preview.call_count, 6)
                self.assertTrue(preview.call_args.args[2])
                uploader.return_value = other
                app.run()
                self.assertNotIn("vehicle_review", app.session_state)
                self.assertNotIn("recognition", app.session_state)
                self.assertNotIn("second_view", app.session_state)
                next(r for r in app.radio if r.label == "Ảnh chiếc xe").set_value("Chụp trực tiếp").run()
                next(b for b in app.button if b.label == "Phân tích ảnh").click().run()
                self.assertGreater(camera.call_count, 0)
                self.assertFalse(app.exception)

    def test_recognition_upload_and_multiple_vehicle_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/processed") as directory:
            run = Path(directory)
            (run / "model.json").write_text("{}")
            uploaded = io.BytesIO()
            Image.new("RGB", (40, 30), "blue").save(uploaded, format="PNG")
            engine = Mock()
            multiple = dict(status="multiple_vehicles", message="Chọn chiếc xe", detections=[{"box": [0, 0, 20, 30]}, {"box": [20, 0, 40, 30]}])
            result = dict(status="tesla_candidate", message="Ứng viên Tesla", candidates=[{"model": "Model 3", "ranking_score": .8}], note="Điểm chưa hiệu chuẩn", explanation=["Đối chiếu ảnh train"], visual_hypotheses=[], references=[])
            engine.analyze.return_value = (multiple, None, None)
            with patch.dict("os.environ", {"CARVISION_VISION_RUN": str(run)}), patch("streamlit.file_uploader", return_value=uploaded), patch("src.recognize_vehicle.Recognizer", return_value=engine):
                app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
                app.sidebar.radio[0].set_value("Nhận diện ảnh").run()
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox), 1)
                engine.analyze.return_value = (result, None, None)
                app.selectbox[0].set_value(1)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(engine.analyze.call_args.args[1], 1)
                self.assertEqual(len(app.get("download_button")), 1)

    def test_cloud_missing_configuration_and_single_image(self):
        output = io.BytesIO()
        Image.new("RGB", (40, 30), "blue").save(output, format="PNG")
        data = output.getvalue()
        row = dict(dataset="test-v1", relative_path="model_3/example.png", class_id="model_3",
                   decision="keep", sha256=hashlib.sha256(data).hexdigest(), purpose="evaluation_only",
                   split="unsplit", license="CC0", provenance={"artist_html": "Test artist"})
        app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
        app.sidebar.radio[0].set_value("Supabase").run()
        with patch("src.cloud_data.CloudData.__init__", return_value=None), patch("src.cloud_data.CloudData.catalog", return_value=[row]), patch("src.cloud_data.CloudData.image", return_value=data) as image:
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("image")), 1)
            app.run()
            self.assertEqual(image.call_count, 1)
        app.button[0].click()
        with patch("src.cloud_data.CloudData.__init__", side_effect=ValueError("Missing configuration")):
            app.run()
            self.assertTrue(app.error)
            self.assertFalse(app.exception)

    def test_empty_state_and_outside_path(self):
        app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
        app.sidebar.radio[0].set_value("Trên máy").run()
        self.assertFalse(app.exception)
        app.sidebar.text_input[0].set_value("../../outside")
        app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)

    def test_real_scan_and_downloads(self):
        raw = ROOT / "data/raw"
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=raw) as temporary:
            dataset = Path(temporary)
            (dataset / "model_3").mkdir()
            Image.new("RGB", (40, 30), "blue").save(dataset / "model_3/test.png")
            app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
            app.sidebar.radio[0].set_value("Trên máy").run()
            app.sidebar.text_input[0].set_value(dataset.name)
            app.sidebar.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "1")
            self.assertEqual(app.metric[1].value, "1")
            self.assertEqual(len(app.get("download_button")), 2)


if __name__ == "__main__":
    unittest.main()
