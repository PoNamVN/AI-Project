const { existsSync } = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function findPython(root, platform = process.platform) {
  const executable = platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python'];
  return ['.venv-vision', '.kickoff-build/vision-v3', '.venv']
    .map(directory => path.join(root, directory, ...executable))
    .find(existsSync);
}

if (require.main === module) {
  const root = path.resolve(__dirname, '..');
  const python = findPython(root);
  if (!python) {
    console.error('Chưa có môi trường Python. Làm theo README.md mục 5 hoặc docs/vision_recognition.md để cài môi trường nhận diện.');
    process.exit(1);
  }
  console.log(`CarVision AI · Python: ${path.relative(root, python)}`);
  const args = process.argv[2] === '--check'
    ? ['-m', 'src.check_environment']
    : ['-m', 'streamlit', 'run', 'app/app.py', ...process.argv.slice(2)];
  const result = spawnSync(python, args, {
    cwd: root,
    stdio: 'inherit',
  });
  if (result.error) console.error(`Không chạy được Python: ${result.error.message}`);
  process.exit(result.status ?? (result.signal === 'SIGINT' ? 130 : 1));
}

module.exports = { findPython };
