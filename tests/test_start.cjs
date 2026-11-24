const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { findPython } = require('../scripts/start.cjs');

test('selects an existing environment and prefers vision over the audit environment', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'carvision-start-'));
  const python = directory => path.join(root, directory, 'Scripts', 'python.exe');
  const add = directory => {
    const file = python(directory);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, '');
    return file;
  };
  try {
    assert.equal(findPython(root, 'win32'), undefined);
    const audit = add('.venv');
    assert.equal(findPython(root, 'win32'), audit);
    const localVision = add('.kickoff-build/vision-v3');
    assert.equal(findPython(root, 'win32'), localVision);
    const sharedVision = add('.venv-vision');
    assert.equal(findPython(root, 'win32'), sharedVision);
    const unix = path.join(root, '.venv', 'bin', 'python');
    mkdirSync(path.dirname(unix), { recursive: true });
    writeFileSync(unix, '');
    assert.equal(findPython(root, 'linux'), unix);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
