"""Downgrade projection executes the real standalone export tool."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/export_legacy_attachments.py'
spec = importlib.util.spec_from_file_location('export_legacy_attachments', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_export_preserves_files_and_pending_without_overwriting(tmp_path):
    item = {'id': 'a', 'path': '/uploads/s/a', 'size': 3}
    envelope = {'version': 2, 'sessions': {
        's': {'attachments': [item], 'pending': False, 'revision': 4, 'uploads': {}},
        'legacy': {'attachments': [], 'pending': True, 'revision': 1, 'uploads': {}},
        'empty': {'attachments': [], 'pending': False, 'revision': 9, 'uploads': {}},
    }}
    backup = tmp_path / 'v2.json'
    backup.write_text(json.dumps(envelope))
    out = tmp_path / 'export'
    subprocess.run([sys.executable, str(SCRIPT), str(backup), str(out)], check=True)
    assert json.loads((out / 'staged_attachments.json').read_text()) == {'s': [item], 'legacy': [], 'empty': []}
    assert json.loads((out / 'pending_attachments.json').read_text()) == ['legacy', 's']
    assert json.loads(backup.read_text()) == envelope
    assert subprocess.run([sys.executable, str(SCRIPT), str(backup), str(out)], capture_output=True).returncode != 0
    assert json.loads((out / 'staged_attachments.json').read_text())['s'] == [item]


@pytest.mark.parametrize('intents', [[{'old_session_id': 'old', 'new_session_id': 'new'}], 'invalid'])
def test_export_refuses_outstanding_or_unverifiable_remaps_without_output(tmp_path, intents):
    backup = tmp_path / 'staged_attachments.json'
    backup.write_text(json.dumps({'version': 2, 'sessions': {}}))
    snapshot = tmp_path / 'session_snapshot.json'
    snapshot.write_text(json.dumps({'remap_intents': intents}))
    before = snapshot.read_bytes()
    out = tmp_path / 'export'
    result = subprocess.run([sys.executable, str(SCRIPT), str(backup), str(out)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'remap_intents' in result.stderr and 'before downgrade' in result.stderr
    assert not out.exists()
    assert snapshot.read_bytes() == before


def test_export_accepts_settled_snapshot(tmp_path):
    backup = tmp_path / 'staged_attachments.json'
    backup.write_text(json.dumps({'version': 2, 'sessions': {}}))
    (tmp_path / 'session_snapshot.json').write_text(json.dumps({'remap_intents': []}))
    out = tmp_path / 'export'
    subprocess.run([sys.executable, str(SCRIPT), str(backup), str(out)], check=True)
    assert json.loads((out / 'staged_attachments.json').read_text()) == {}


@pytest.mark.parametrize('value', [{}, {'version': 1, 'sessions': {}}, {'version': 2, 'sessions': {'s': {'attachments': [], 'pending': 'false'}}}])
def test_rejects_unknown_or_invalid_envelope(value):
    with pytest.raises(ValueError):
        module.legacy_projection(value)
