"""Public Streamlit entry point: cloud required, internal tools always disabled."""

import os
from pathlib import Path
import re
import subprocess
import sys

from src.cloud_data import CloudData
from src.check_environment import check


def command():
    CloudData()
    if any(not ok for _, ok, required in check() if required):
        raise ValueError("Thiếu thư viện/model/cache; kiểm tra bước build trước khi khởi động.")
    port = os.environ.get("PORT", "10000")
    if not re.fullmatch(r"[0-9]{1,5}", port) or not 1 <= int(port) <= 65535:
        raise ValueError("Cổng máy chủ không hợp lệ.")
    os.environ["CARVISION_ENABLE_DEV_TOOLS"] = "0"
    return [sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.address=0.0.0.0",
            f"--server.port={port}", "--server.headless=true", "--server.maxUploadSize=30",
            "--browser.gatherUsageStats=false", "--client.showErrorDetails=none", "--client.toolbarMode=viewer"]


if __name__ == "__main__":
    try:
        args = command()
    except (ValueError, OSError):
        raise SystemExit("Ứng dụng chưa đủ cấu hình công khai. Kiểm tra runtime và cấu hình Supabase riêng trên máy chủ.") from None
    os.chdir(Path(__file__).resolve().parents[1])
    if os.name == "nt":
        raise SystemExit(subprocess.call(args))
    os.execv(sys.executable, args)
