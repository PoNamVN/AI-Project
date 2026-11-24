const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { findPython, loadServerConfig, compactReady } = require('../scripts/start.cjs');

test('loads only cloud config and preserves explicit deployment settings', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'carvision-env-'));
  try {
    const empty = {};
    loadServerConfig(root, empty);
    assert.deepEqual(empty, {});
    writeFileSync(path.join(root, '.env'), 'CARVISION_SUPABASE_URL=https://example.supabase.co\nCARVISION_SUPABASE_KEY="fixture-key-$HOME"\nPYTHONPATH=unsafe-fixture\nNODE_OPTIONS=unsafe-fixture\n');
    const env = { CARVISION_SUPABASE_URL: 'deployment-url' };
    loadServerConfig(root, env);
    assert.deepEqual(env, { CARVISION_SUPABASE_URL: 'deployment-url', CARVISION_SUPABASE_KEY: 'fixture-key-$HOME' });
    assert.equal(env.PYTHONPATH, undefined);
    assert.equal(env.NODE_OPTIONS, undefined);
    writeFileSync(path.join(root, '.env'), 'x'.repeat(16385));
    assert.throws(() => loadServerConfig(root, {}), /Invalid server config/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

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

test('compact mode requires its own environment and a completed validation record', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'carvision-compact-'));
  try {
    assert.equal(compactReady(root), false);
    assert.equal(findPython(root, 'win32', 'onnx'), undefined);
    const file = path.join(root, '.venv-onnx', 'Scripts', 'python.exe');
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, '');
    assert.equal(findPython(root, 'win32', 'onnx'), file);
    const runtime = path.join(root, 'data', 'processed', 'vision-onnx-v31');
    mkdirSync(runtime, { recursive: true });
    writeFileSync(path.join(runtime, 'runtime.json'), '{}');
    const proof = path.join(runtime, 'validation.json');
    writeFileSync(proof, '{"passed":false}');
    assert.equal(compactReady(root), false);
    writeFileSync(proof, '{"passed":true}');
    assert.equal(compactReady(root), true);
    writeFileSync(proof, 'malformed');
    assert.equal(compactReady(root), false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
