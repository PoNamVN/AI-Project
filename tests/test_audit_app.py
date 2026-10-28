"""UI checks use synthetic images only, never project dataset evidence."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_empty_state_and_outside_path(self):
        app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=30)
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
            app.sidebar.text_input[0].set_value(dataset.name)
            app.sidebar.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "1")
            self.assertEqual(app.metric[1].value, "1")
            self.assertEqual(len(app.get("download_button")), 2)


if __name__ == "__main__":
    unittest.main()
