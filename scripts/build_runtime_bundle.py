"""Export only the existing V3.1 runtime; never train or include credentials."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'data/processed/vision-recognition-v3-brand'
DETECTOR = '.kickoff-build/torch/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
HISTORY = ('tesla-expanded-v2/splits.csv', 'week4-newcheck-v1/review.csv',
           'non-tesla-gate-review-v1/review.csv', 'non-tesla-brand-training-v3/review.csv')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_files(root):
    root = Path(root).resolve()
    files = {}

    def add(relative, maximum):
        parts = PurePosixPath(relative).parts
        if not parts or '..' in parts or ':' in relative or '\\' in relative or relative.startswith('/'):
            raise ValueError('Invalid runtime path')
        candidate = root / relative
        path = candidate.resolve()
        if candidate.is_symlink() or path != candidate.absolute():
            raise ValueError('Runtime inputs must be regular files, not symbolic links')
        if not path.is_relative_to(root) or not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise ValueError(f'Missing or invalid runtime file: {relative}')
        files[relative] = path

    add(MODEL + '/model.json', 5_000_000)
    add(MODEL + '/classifier.npz', 20_000_000)
    metadata = json.loads(files[MODEL + '/model.json'].read_text(encoding='utf-8'))
    for key, expected in (('dino', 'facebook/dinov2-small'), ('siglip', 'google/siglip2-base-patch16-224')):
        name, revision = metadata[key]
        if name != expected or not re.fullmatch('[a-f0-9]{40}', revision):
            raise ValueError('Invalid encoder checkpoint')
        snapshot = f'.kickoff-build/huggingface/models--{name.replace("/", "--")}/snapshots/{revision}'
        names = ['config.json', 'preprocessor_config.json', 'model.safetensors']
        if key == 'siglip':
            names += ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
        for filename in names:
            add(snapshot + '/' + filename, 2_000_000_000)
    add(DETECTOR, 50_000_000)
    for name in HISTORY:
        add('data/processed/' + name, 10_000_000)
    for pattern in ('*/review.csv', '*/splits.csv'):
        for manifest in (root / 'data/processed').glob(pattern):
            add(manifest.relative_to(root).as_posix(), 10_000_000)
    records = metadata['train_records']
    if not records:
        raise ValueError('No training references in model metadata')
    for record in records:
        relative = record['source_path']
        parts = PurePosixPath(relative).parts
        if (len(parts) != 3 or parts[1] != record['class_id']
                or parts[1] not in ('model_s', 'model_3', 'model_x', 'model_y', 'cybertruck')
                or PurePosixPath(relative).suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
                or not re.fullmatch('[a-f0-9]{64}', record['sha256'])
                or not all(record.get(k) for k in ('source_url', 'license', 'artist_html', 'license_url'))):
            raise ValueError('Invalid reference provenance')
        add('data/raw/' + relative, 30_000_000)
        if sha256(files['data/raw/' + relative]) != record['sha256']:
            raise ValueError('Reference checksum changed')
    return files


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    checksum = output.with_suffix(output.suffix + '.sha256')
    if output.exists() or checksum.exists():
        raise ValueError('Choose a new output; never overwrite a bundle')
    if not output.is_relative_to(root / '.kickoff-build') or output.suffix != '.zip':
        raise ValueError('Output must be a new ZIP under .kickoff-build')
    files = runtime_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='runtime-', dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        records = []
        with zipfile.ZipFile(staged, 'x', compression=zipfile.ZIP_STORED) as archive:
            for relative, path in sorted(files.items()):
                digest = hashlib.sha256()
                size = 0
                with path.open('rb') as source, archive.open(relative, 'w', force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                        size += len(chunk)
                        target.write(chunk)
                records.append({'path': relative, 'bytes': size, 'sha256': digest.hexdigest()})
            for name in ('runtime_notices.md', 'licenses/Apache-2.0.txt', 'licenses/Torchvision-BSD.txt'):
                source = root / 'docs' / name
                if not source.is_file() or source.is_symlink() or source.resolve() != source.absolute() or source.stat().st_size > 200_000:
                    raise ValueError(f'Missing notice: docs/{name}')
                data = source.read_bytes()
                relative = '.kickoff-build/runtime-licenses/' + name
                archive.writestr(relative, data)
                records.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            archive.writestr('.kickoff-build/runtime-bundle-manifest.json', json.dumps({
                'schema_version': 1, 'model': MODEL, 'files': records,
                'note': 'Existing development runtime and references; no new labels or accuracy.'
            }, ensure_ascii=False, indent=2))
        staged_checksum = Path(temporary) / checksum.name
        staged_checksum.write_text(f'{sha256(staged)}  {output.name}\n', encoding='ascii')
        staged.rename(output)
        staged_checksum.rename(checksum)
    return {'files': len(records), 'bytes': output.stat().st_size, 'bundle': str(output), 'sha256_file': str(checksum)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(ROOT, args.output), ensure_ascii=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + '\n')
