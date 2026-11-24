const { existsSync, lstatSync, readFileSync } = require('node:fs');
const { parseEnv } = require('node:util');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function findPython(root, platform = process.platform, backend = 'torch') {
  const executable = platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python'];
  const directories = backend === 'onnx' ? ['.venv-onnx'] : ['.venv-vision', '.kickoff-build/vision-v3', '.venv'];
  return directories
    .map(directory => path.join(root, directory, ...executable))
    .find(existsSync);
}

function compactReady(root) {
  const directory = path.join(root, 'data', 'processed', 'vision-onnx-v31');
  const proof = path.join(directory, 'validation.json');
  try {
    const info = lstatSync(proof);
    return info.isFile() && !info.isSymbolicLink() && info.size <= 64000
      && existsSync(path.join(directory, 'runtime.json')) && JSON.parse(readFileSync(proof, 'utf8')).passed === true;
  } catch { return false; }
}

function loadServerConfig(root, env = process.env) {
  const file = path.join(root, '.env');
  if (!existsSync(file)) return;
  const info = lstatSync(file);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 16384) throw new Error('Invalid server config');
  if (typeof parseEnv !== 'function') throw new Error('Node 20.12 or newer required');
  const values = parseEnv(readFileSync(file, 'utf8'));
  for (const key of ['CARVISION_SUPABASE_URL', 'CARVISION_SUPABASE_KEY']) {
    if (env[key] === undefined && values[key] !== undefined) env[key] = values[key];
  }
}

if (require.main === module) {
  const root = path.resolve(__dirname, '..');
  try { loadServerConfig(root); } catch {
    console.error('Không đọc được cấu hình .env. Dùng Node.js 20.12 trở lên và kiểm tra tệp cấu hình máy chủ; không gửi khóa vào chat.');
    process.exit(1);
  }
  const mode = process.argv[2];
  const backend = mode === '--torch' ? 'torch' : mode === '--lite' ? 'onnx'
    : process.env.CARVISION_INFERENCE_BACKEND || (compactReady(root) && findPython(root, process.platform, 'onnx') ? 'onnx' : 'torch');
  if (!['torch', 'onnx'].includes(backend)) {
    console.error('CARVISION_INFERENCE_BACKEND chỉ nhận torch hoặc onnx.');
    process.exit(1);
  }
  process.env.CARVISION_INFERENCE_BACKEND = backend;
  const python = findPython(root, process.platform, backend);
  if (!python) {
    console.error('Chưa có môi trường Python. Làm theo README.md mục 5 hoặc docs/vision_recognition.md để cài môi trường nhận diện.');
    process.exit(1);
  }
  console.log(`CarVision AI · Python: ${path.relative(root, python)}`);
  if (mode === '--dev-tools') process.env.CARVISION_ENABLE_DEV_TOOLS = '1';
  const args = mode === '--check' ? ['-m', 'src.check_environment']
    : mode === '--setup-cloud' ? ['-m', 'src.setup_cloud']
    : mode === '--check-cloud' ? ['-m', 'src.setup_cloud', '--check']
    : mode === '--prepare-previews' ? ['-m', 'src.prepare_cloud_previews', ...process.argv.slice(3)]
    : ['-m', 'streamlit', 'run', 'app/app.py', ...process.argv.slice(['--dev-tools', '--lite', '--torch'].includes(mode) ? 3 : 2)];
  const result = spawnSync(python, args, {
    cwd: root,
    stdio: 'inherit',
  });
  if (result.error) console.error(`Không chạy được Python: ${result.error.message}`);
  process.exit(result.status ?? (result.signal === 'SIGINT' ? 130 : 1));
}

module.exports = { findPython, loadServerConfig, compactReady };
