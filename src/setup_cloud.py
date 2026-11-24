"""Owner-only setup: hidden key entry, verify one private reference, then save."""

import argparse
import csv
import getpass
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from src.cloud_data import CloudData
from src.data_audit import MODELS
from src.vision_features import open_image

ROOT = Path(__file__).resolve().parents[1]
PROJECT_URL = 'https://bnfqwqpqxoovoxnwunnx.supabase.co'


def verify_cloud(root=ROOT, client=None):
    root = Path(root).resolve()
    run = Path(os.environ.get('CARVISION_VISION_RUN', str(root / 'data/processed/vision-recognition-v3-brand'))).resolve()
    config = run / 'model.json'
    if not run.is_relative_to(root / 'data/processed') or not config.is_file() or config.stat().st_size > 5_000_000:
        raise ValueError('Cần gói runtime/model hợp lệ trên máy trước khi kiểm tra ảnh cloud.')
    records = json.loads(config.read_text(encoding='utf-8'))['train_records']
    if not records or not isinstance(records, list):
        raise ValueError('Model chưa có danh sách ảnh đối chiếu.')
    record = records[0]
    if record.get('class_id') not in MODELS or not re.fullmatch('[a-f0-9]{64}', record.get('sha256', '')):
        raise ValueError('Thông tin ảnh đối chiếu không hợp lệ.')
    cloud = CloudData() if client is None else client
    data = cloud.reference_image(record['sha256'], record['class_id'])
    with open_image(data) as image:
        return {'class_id': record['class_id'], 'bytes': len(data), 'width': image.width, 'height': image.height}


def save_config(root, key):
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{20,4096}', key):
        raise ValueError('Khóa không hợp lệ; không dùng khoảng trắng hoặc xuống dòng.')
    path = Path(root).resolve() / '.env'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            if os.name == 'nt':
                identity = subprocess.run(['whoami', '/user', '/fo', 'csv', '/nh'], capture_output=True, text=True, check=True)
                sid = next(csv.reader(io.StringIO(identity.stdout)))[1]
                if not re.fullmatch(r'S-1-[0-9-]+', sid):
                    raise ValueError('Không xác định được người dùng Windows.')
                subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r', f'*{sid}:(F)',
                                '/grant:r', '*S-1-5-18:(F)'], capture_output=True, check=True)
            handle.write(f'CARVISION_SUPABASE_URL={PROJECT_URL}\nCARVISION_SUPABASE_KEY={key}\n')
    except (OSError, ValueError, subprocess.SubprocessError, StopIteration, IndexError):
        path.unlink(missing_ok=True)
        raise ValueError('Chưa lưu cấu hình vì không bảo vệ được tệp. Tệp cũ không bị ghi đè.') from None


def setup(root=ROOT):
    if (Path(root) / '.env').exists():
        raise ValueError('Đã có .env; giữ nguyên tệp cũ. Dùng npm run check:cloud để kiểm tra.')
    if not sys.stdin.isatty():
        raise ValueError('Mở terminal của bạn và chạy npm run setup:cloud để nhập khóa ẩn; không gửi khóa qua chat.')
    key = getpass.getpass('Dan secret key cua CarVision AI (se khong hien ky tu): ').strip()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{20,4096}', key):
        raise ValueError('Khóa không hợp lệ hoặc bị trống; chưa lưu cấu hình.')
    report = verify_cloud(root, CloudData(PROJECT_URL, key))
    save_config(root, key)
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Đọc thử một ảnh, không nhập hoặc lưu khóa')
    args = parser.parse_args()
    try:
        report = verify_cloud() if args.check else setup()
        print('Đã tải một ảnh Supabase và kiểm tra checksum/định dạng: ' + json.dumps(report, ensure_ascii=False))
        if not args.check:
            print('Đã lưu cấu hình riêng trên máy. Khởi động lại app bằng npm start để dùng ảnh cloud.')
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, 'Đã hủy. Chưa hoàn thành thiết lập.\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
