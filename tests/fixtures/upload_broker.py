"""Idle container-only Pi-shaped broker; record every mutating command."""
import json
import os
from pathlib import Path
import socket
import time

home = Path.home()
socks = home / '.local/share/codoxear/socks'
socks.mkdir(parents=True, exist_ok=True)
path = socks / 'activity-peek.sock'
log = home / 'upload-fixture.jsonl'
log.write_text(json.dumps({'type': 'session', 'version': 3, 'id': 'upload-fixture', 'cwd': '/workspace', 'timestamp': '2026-10-26T10:00:00Z'}) + '\n')
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(path))
server.listen(16)
path.with_suffix('.json').write_text(json.dumps({
    'session_id': 'activity-peek', 'agent_backend': 'pi', 'owner': 'web',
    'broker_pid': os.getpid(), 'codex_pid': os.getpid(), 'cwd': '/workspace',
    'start_ts': time.time(), 'updated_ts': time.time(), 'log_path': str(log),
    'model': 'fixture', 'control_protocol_version': 2,
    'control_capabilities': {'sync_send': True, 'key_write_errors': True},
}))
while True:
    conn, _ = server.accept()
    with conn:
        raw = b''
        while not raw.endswith(b'\n'):
            chunk = conn.recv(65536)
            if not chunk:
                break
            raw += chunk
        request = json.loads(raw)
        if request.get('cmd') == 'state':
            response = {'busy': False, 'turn_open': False, 'queue_len': 0, 'token': None}
        else:
            with (home / 'broker-mutations.jsonl').open('a') as stream:
                stream.write(json.dumps(request) + '\n')
            response = {'error': 'mutation forbidden in upload verification'}
        conn.sendall(json.dumps(response).encode() + b'\n')
