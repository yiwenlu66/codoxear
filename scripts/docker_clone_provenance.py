#!/usr/bin/env python3
"""Fail before pytest if the isolated suite would import a different checkout."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
expected = sys.argv[1]
actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
if Path.cwd().resolve() != root or actual != expected:
    raise SystemExit(f'clone identity mismatch: cwd={Path.cwd()} root={root} HEAD={actual} expected={expected}')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'tests'))
modules = {}
for name in ['codoxear', 'frontend_module_loader', 'test_path_completion']:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f'{name} imported outside verified clone: {path}')
    modules[name] = str(path)

loader = importlib.import_module('frontend_module_loader')
paths = importlib.import_module('test_path_completion')
source_hashes = []
for name in ['app_file_helpers.js', 'app_path_completion.js', 'app_session_state.js', 'app_session_catalog.js', 'app_composer.js']:
    raw = root / 'codoxear' / 'static' / name
    projected = loader.module_path(name)
    source_hashes.append({'source': str(raw), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                          'vm_projection': str(projected), 'vm_sha256': hashlib.sha256(projected.read_bytes()).hexdigest()})
program = 'const SOURCES = ' + json.dumps(paths.SOURCES) + ';\n' + paths.HARNESS
print(json.dumps({'expected_head': expected, 'actual_head': actual, 'clone_root': str(root),
                  'python': sys.executable, 'node': shutil.which('node'), 'modules': modules,
                  'vm_sources_count': len(paths.SOURCES), 'vm_program_prefix_bytes': len(program.encode()),
                  'source_hashes': source_hashes}, indent=2), flush=True)
