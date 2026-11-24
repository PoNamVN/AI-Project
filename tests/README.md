# Kiểm thử

Môi trường nhận diện có `requirements-demo.txt` và `requirements-vision.txt`:

```powershell
.\.venv-vision\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
node --test tests/test_start.cjs
```

Các kiểm thử TensorFlow có thể bị bỏ qua trong môi trường vision. Phải chạy riêng trong môi trường
đã cài `requirements-baseline.txt` trước khi công bố thay đổi CNN/MobileNetV2:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv-ml\Scripts\python.exe -m unittest discover -s tests -p "test_train_baseline.py" -v
```

Hai môi trường tách riêng để không ép TensorFlow và PyTorch dùng chung phiên bản NumPy.
Kiểm thử dùng ảnh tổng hợp và giả lập nơi cần; không đo accuracy đường phố và không thay thế thử camera/ảnh cloud thật.
Kiểm tra bàn giao model/cache và chạy thử ảnh thật theo [hướng dẫn máy mới](../docs/team_setup.md).
