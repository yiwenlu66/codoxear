"""Executable attachment authority, interruption and binary protocol contracts."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from codoxear.control_routes import ControlRouteDeps, handle_control_get_route, handle_control_post_route
from codoxear.file_upload import stage_uploaded_file
from codoxear.server import _match_session_route
from codoxear.server_http import read_body
from codoxear.session_pending_state import SessionPendingStateCoordinator
from codoxear.session_store import SessionStore, SessionStorePaths


def make_store(root):
    return SessionStore(
        paths=SessionStorePaths(**{name: root / (name + '.json') for name in (
            'aliases', 'sidebar_meta', 'hidden_sessions', 'files', 'queues', 'pending_attachments',
            'commit_unknown_sends', 'recent_cwds', 'unattended', 'staged_attachments')}, uploads_root=root / 'uploads', drafts=root / 'drafts.json'),
        file_history_max=5, recent_cwd_max=5, unattended_default_idle_minutes=5,
        unattended_default_max_injections=5, clean_alias=lambda x: x, clean_priority_offset=lambda x: x,
        clean_snooze_until=lambda x: x, clean_dependency_session_id=lambda x: x,
        clean_recent_cwd=lambda x: x, clean_commit_unknown_send_record=lambda x: x,
    )


def coordinator(store):
    sessions = {'s1': SimpleNamespace(pending_attachment=False), 's2': SimpleNamespace(pending_attachment=False)}
    return SessionPendingStateCoordinator(
        lock=threading.RLock(), sessions=lambda: sessions, store=lambda: store,
        pending_attachment_ids=lambda: store.pending_attachment_ids,
        set_pending_attachment_ids=lambda ids: setattr(store, 'pending_attachment_ids', ids),
        commit_unknown_sends=lambda: store.commit_unknown_sends, set_commit_unknown_sends=lambda x: None,
        mark_queue_orphan_recovery_locked=lambda sid: False, save_pending_attachments=lambda: None,
        save_commit_unknown_sends=lambda: None, save_queues=lambda: None, now=lambda: 100.0,
        commit_unknown_orphan_prune_seconds=100,
    )


def finish(c, uid='upload-1', raw=b'hello', sid='s1'):
    return c.finish_upload(sid, uid, filename='hello.txt', raw=raw, stage=lambda: stage_uploaded_file(
        sid, 'hello.txt', raw, upload_dir=c.store().paths.uploads_root, now_fn=lambda: 100, max_bytes=16 * 1024 * 1024))


def test_duplicate_concurrency_response_lost_restart_and_retirement(tmp_path):
    store = make_store(tmp_path)
    c = coordinator(store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5), range(8)))
    assert sum(result['writer'] for result in claims) == 1
    assert all(result['state'] == 'pending' for result in claims)
    committed = finish(c)  # Imagine the response is lost here.
    assert committed['state'] == 'committed' and committed['active']
    restored = make_store(tmp_path)
    restored.load_persistent_state()
    c2 = coordinator(restored)
    assert c2.upload_status('s1', 'upload-1') == committed
    assert not c2.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)['writer']
    assert finish(c2) == committed
    assert len(list((tmp_path / 'uploads' / 's1').iterdir())) == 1
    cleared = c2.remove_staged_attachment('s1', committed['attachment']['id'])
    assert cleared['attachments_revision'] > committed['attachments_revision']
    assert cleared['attachments'] == []
    assert finish(c2)['active'] is False
    assert not list((tmp_path / 'uploads' / 's1').iterdir())
    restored.load_persistent_state()
    assert coordinator(restored).upload_status('s1', 'upload-1')['active'] is False


def test_identity_conflict_and_pending_restart(tmp_path):
    c = coordinator(make_store(tmp_path))
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    with pytest.raises(ValueError, match='incompatible'):
        c.begin_upload('s1', 'upload-1', filename='other.txt', size=5)
    restored = make_store(tmp_path)
    restored.load_persistent_state()
    restarted = coordinator(restored)
    assert restarted.upload_status('s1', 'upload-1')['state'] == 'failed'
    assert restarted.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)['writer']
    finish(restarted)
    with pytest.raises(ValueError, match='incompatible'):
        finish(restarted, raw=b'other')


def test_cancel_before_body_after_commit_and_clear_pending(tmp_path):
    c = coordinator(make_store(tmp_path))
    c.cancel_upload('s1', 'pre-cancel')
    assert not c.begin_upload('s1', 'pre-cancel', filename='hello.txt', size=5)['writer']
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    c.clear_staged_attachments('s1')
    assert finish(c)['state'] == 'cancelled'
    assert not (tmp_path / 'uploads').exists()
    c.begin_upload('s1', 'upload-2', filename='hello.txt', size=5)
    committed = finish(c, 'upload-2')
    cancelled = c.cancel_upload('s1', 'upload-2')
    assert cancelled['state'] == 'committed' and not cancelled['active']
    assert cancelled['attachments_revision'] > committed['attachments_revision']


def test_send_consumes_captured_ids_only_retains_bytes(tmp_path):
    c = coordinator(make_store(tmp_path))
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    first = finish(c)
    c.begin_upload('s1', 'upload-2', filename='hello.txt', size=5)
    second = finish(c, 'upload-2')
    consumed = c.clear_staged_attachments('s1', delete_files=False, attachment_ids=[first['attachment']['id']])
    assert consumed['attachments'] == [second['attachment']]
    assert consumed['pending_attachment'] is True
    assert len(list((tmp_path / 'uploads' / 's1').iterdir())) == 2
    assert finish(c)['active'] is False


def test_atomic_failure_no_false_commit_and_stale_save_cannot_rollback(tmp_path):
    store = make_store(tmp_path)
    c = coordinator(store)
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    with patch('codoxear.session_store.durable_write_resource', side_effect=OSError('disk full')):
        with pytest.raises(OSError):
            finish(c)
    assert c.upload_status('s1', 'upload-1')['state'] == 'pending'
    assert not list((tmp_path / 'uploads' / 's1').iterdir())
    committed = finish(c)
    stale = dict(store.staged_attachments)
    deleted = c.clear_staged_attachments('s1')
    store.save_staged_attachments(stale)
    restored = make_store(tmp_path)
    restored.load_persistent_state()
    snapshot = restored.attachment_snapshot('s1')
    assert snapshot['attachments'] == []
    assert snapshot['attachments_revision'] == deleted['attachments_revision'] > committed['attachments_revision']


def test_legacy_load_empty_tombstone_pending_and_remap(tmp_path):
    store = make_store(tmp_path)
    c = coordinator(store)
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    committed = finish(c)
    store.remap_attachment_resource('s1', 's2')
    moved = c.upload_status('s2', 'upload-1')
    assert moved['active'] and moved['attachment'] == committed['attachment']
    assert moved['attachments_revision'] > committed['attachments_revision']
    assert c.upload_status('s1', 'upload-1')['active'] is False
    store.clear_deleted_session_state('s1')
    assert Path(store.staged_attachments['s2'][0]['path']).read_bytes() == b'hello'
    c.remove_staged_attachment('s2', moved['attachment']['id'])
    # The old pending flag file must never resurrect a versioned tombstone.
    store.paths.pending_attachments.write_text(json.dumps(['s1', 's2']))
    restored = make_store(tmp_path)
    restored.load_persistent_state()
    assert restored.pending_attachment_ids == set()
    legacy_root = tmp_path / 'legacy'
    legacy_root.mkdir()
    legacy = make_store(legacy_root)
    entry = {'id': 'a1', 'filename': 'old', 'path': '/old', 'size': 3, 'created_ts': 1}
    legacy.paths.staged_attachments.write_text(json.dumps({'s1': [entry]}))
    legacy.paths.pending_attachments.write_text(json.dumps(['s2']))
    legacy.load_persistent_state()
    assert legacy.pending_attachment_ids == {'s1', 's2'}
    assert legacy.attachment_snapshot('s1')['attachments'][0]['id'] == 'a1'
    assert legacy.attachment_snapshot('s2')['pending_attachment']


class Handler:
    def __init__(self, raw, headers, path='/api/sessions/s1/inject_file'):
        self.headers = headers
        self.rfile = io.BytesIO(raw)
        self.path = path
    def _unauthorized(self):
        raise AssertionError('unexpected auth failure')


def route_deps(c, responses):
    return ControlRouteDeps(
        require_auth=lambda h: True, json_response=lambda h, status, result: responses.append((status, result)),
        read_body=read_body, read_json_body=lambda handler, **kwargs: json.loads(read_body(handler, limit=kwargs.get("limit", 1024))), attach_upload_body_max_bytes=1024,
        attach_upload_max_bytes=512, stage_uploaded_file=lambda sid, name, raw: stage_uploaded_file(
            sid, name, raw, upload_dir=c.store().paths.uploads_root, now_fn=lambda: 100, max_bytes=512),
        clean_unattended_cooldown_minutes=int, clean_unattended_remaining_injections=int,
        session_not_ready_error=RuntimeError, session_injection_error=RuntimeError,
        session_commit_unknown_error=RuntimeError,
    )


@pytest.mark.parametrize('binary', [True, False])
def test_raw_and_legacy_identified_route_status_no_paths(tmp_path, binary):
    c = coordinator(make_store(tmp_path))
    manager = SimpleNamespace(**{name: getattr(c, name) for name in (
        'upload_status', 'begin_upload', 'finish_upload', 'fail_upload', 'cancel_upload', 'add_staged_attachment')},
        attachment_staging_ready=lambda sid: True)
    responses = []
    deps = route_deps(c, responses)
    raw = b'hello'
    if binary:
        body = raw
        headers = {'Content-Type': 'application/octet-stream', 'Content-Length': '5',
                   'X-Codoxear-Upload-Id': 'upload-1', 'X-Codoxear-Filename': 'hello.txt'}
    else:
        body = json.dumps({'filename': 'hello.txt', 'data_b64': base64.b64encode(raw).decode(), 'upload_id': 'upload-1'}).encode()
        headers = {'Content-Type': 'application/json', 'Content-Length': str(len(body))}
    h = Handler(body, headers)
    assert handle_control_post_route(h, path='/api/sessions/s1/inject_file', manager=manager, deps=deps, match_session_route=_match_session_route)
    assert responses[-1][0] == 200
    committed = responses[-1][1]
    assert committed['state'] == 'committed' and committed['active']
    assert 'path' not in committed['attachment']
    h = Handler(b'', {}, '/api/sessions/s1/attachments/upload-status?upload_id=upload-1')
    assert handle_control_get_route(h, path='/api/sessions/s1/attachments/upload-status', manager=manager, deps=deps, match_session_route=_match_session_route)
    assert responses[-1] == (200, committed)
    assert h._codoxear_response_no_store is True
    assert Path(c.store().staged_attachments['s1'][0]['path']).read_bytes() == raw


def test_post_replace_fsync_failure_never_unlinks_published_attachment(tmp_path):
    from codoxear.attachment_resource import AttachmentPersistenceUncertain
    import os
    import stat
    c = coordinator(make_store(tmp_path))
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    original_fsync = os.fsync
    directories = 0
    def fail_second_directory(fd):
        nonlocal directories
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directories += 1
            if directories == 4:  # upload directories succeed; envelope directory fails
                raise OSError('directory sync failed')
        return original_fsync(fd)
    with patch('os.fsync', side_effect=fail_second_directory):
        with pytest.raises(AttachmentPersistenceUncertain):
            finish(c)
    assert c.upload_status('s1', 'upload-1')['state'] == 'committed'
    restored = make_store(tmp_path)
    restored.load_persistent_state()
    assert coordinator(restored).upload_status('s1', 'upload-1')['active']
    assert Path(restored.staged_attachments['s1'][0]['path']).read_bytes() == b'hello'


def test_remap_pending_can_retry_and_failure_rolls_back(tmp_path):
    c = coordinator(make_store(tmp_path))
    c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)
    c.store().remap_attachment_resource('s1', 's2')
    assert c.upload_status('s1', 'upload-1')['state'] == 'cancelled'
    assert c.upload_status('s2', 'upload-1')['state'] == 'failed'
    assert c.begin_upload('s2', 'upload-1', filename='hello.txt', size=5)['writer']
    finish(c, sid='s2')
    with patch('codoxear.session_store.durable_write_resource', side_effect=OSError('no space')):
        with pytest.raises(OSError):
            c.store().remap_attachment_resource('s2', 's3')
    assert c.upload_status('s2', 'upload-1')['active']
    assert c.upload_status('s1', 'upload-1')['state'] == 'cancelled'


def test_http_lost_response_recovers_single_committed_upload(tmp_path):
    """Real socket POST -> lost response -> authoritative GET -> same-ID POST."""
    import http.client
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import socket
    from urllib.parse import urlsplit
    from codoxear.server_http import json_response
    c = coordinator(make_store(tmp_path))
    manager = SimpleNamespace(**{name: getattr(c, name) for name in (
        'upload_status', 'begin_upload', 'finish_upload', 'fail_upload', 'cancel_upload', 'add_staged_attachment')},
        attachment_staging_ready=lambda sid: True)
    lost = threading.Event()
    deps = route_deps(c, [])
    def response(h, status, result):
        if h.headers.get('X-Test-Lose-Response') == '1':
            lost.set()
            h.connection.shutdown(socket.SHUT_RDWR)
            h.connection.close()
            raise BrokenPipeError('response lost')
        json_response(h, status, result, set_auth_cookie=lambda _: None)
    from dataclasses import replace
    deps = replace(deps, json_response=response)
    class HTTPHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):
            pass
        def do_POST(self):
            try:
                handle_control_post_route(self, path=urlsplit(self.path).path, manager=manager,
                                          deps=deps, match_session_route=_match_session_route)
            except BrokenPipeError:
                self.close_connection = True
        def do_GET(self):
            handle_control_get_route(self, path=urlsplit(self.path).path, manager=manager,
                                     deps=deps, match_session_route=_match_session_route)
    server = ThreadingHTTPServer(('127.0.0.1', 0), HTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = None
    try:
        headers = {'Content-Type': 'application/octet-stream', 'X-Codoxear-Upload-Id': 'upload-1',
                   'X-Codoxear-Filename': 'hello.txt', 'X-Test-Lose-Response': '1'}
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request('POST', '/api/sessions/s1/inject_file', b'hello', headers)
        with pytest.raises(http.client.RemoteDisconnected):
            connection.getresponse()
        connection.close()
        assert lost.wait(2)
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request('GET', '/api/sessions/s1/attachments/upload-status?upload_id=upload-1')
        received = connection.getresponse()
        recovered = json.loads(received.read())
        assert received.status == 200 and recovered['state'] == 'committed' and recovered['active']
        headers.pop('X-Test-Lose-Response')
        connection.request('POST', '/api/sessions/s1/inject_file', b'hello', headers)
        received = connection.getresponse()
        assert received.status == 200 and json.loads(received.read()) == recovered
        assert len(list((tmp_path / 'uploads' / 's1').iterdir())) == 1
    finally:
        if connection:
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


def manager_for_routes(c):
    return SimpleNamespace(**{name: getattr(c, name) for name in (
        'upload_status', 'begin_upload', 'finish_upload', 'fail_upload', 'cancel_upload',
        'add_staged_attachment', 'list_staged_attachments')}, attachment_staging_ready=lambda sid: True)


def post_upload(c, *, identified=True, responses=None):
    responses = responses if responses is not None else []
    if identified:
        body = b'hello'
        headers = {'Content-Type': 'application/octet-stream', 'Content-Length': '5',
                   'X-Codoxear-Upload-Id': 'upload-1', 'X-Codoxear-Filename': 'hello.txt'}
    else:
        body = json.dumps({'filename': 'hello.txt', 'data_b64': 'aGVsbG8='}).encode()
        headers = {'Content-Type': 'application/json', 'Content-Length': str(len(body))}
    handle_control_post_route(Handler(body, headers), path='/api/sessions/s1/inject_file',
                             manager=manager_for_routes(c), deps=route_deps(c, responses),
                             match_session_route=_match_session_route)
    return responses[-1]


def fail_next_resource_directory_sync():
    """Fail the actual post-replace fsync, not file write or byte staging."""
    from contextlib import contextmanager
    from codoxear.attachment_resource import durable_write_resource
    import os
    import stat
    real_sync = os.fsync
    pending = True
    def write(path, payload):
        nonlocal pending
        if not pending:
            return durable_write_resource(path, payload)
        pending = False
        def sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('injected metadata directory fsync failure')
            real_sync(fd)
        with patch('os.fsync', side_effect=sync):
            durable_write_resource(path, payload)
    @contextmanager
    def failing():
        with patch('codoxear.session_store.durable_write_resource', side_effect=write):
            yield
    return failing()


def test_legacy_route_post_publication_failure_retains_referenced_bytes(tmp_path):
    c = coordinator(make_store(tmp_path))
    with fail_next_resource_directory_sync():
        status, _ = post_upload(c, identified=False)
    assert status == 503
    current = c.list_staged_attachments('s1')
    assert len(current['attachments']) == 1 and current['pending_attachment']
    restored = make_store(tmp_path)
    restored.load_persistent_state()
    assert restored.attachment_snapshot('s1') == current
    assert Path(restored.staged_attachments['s1'][0]['path']).read_bytes() == b'hello'


def test_legacy_route_pre_publication_failure_unlinks_unreferenced_bytes(tmp_path):
    c = coordinator(make_store(tmp_path))
    with patch('codoxear.session_store.durable_write_resource', side_effect=OSError('no space')):
        assert post_upload(c, identified=False)[0] == 503
    assert not c.list_staged_attachments('s1')['attachments']
    assert not list((tmp_path / 'uploads' / 's1').iterdir())


def test_begin_route_uncertain_publication_is_retryable_without_restart(tmp_path):
    c = coordinator(make_store(tmp_path))
    with fail_next_resource_directory_sync():
        assert post_upload(c)[0] == 503
    status = c.upload_status('s1', 'upload-1')
    assert status['state'] == 'failed' and status['retryable']
    assert not (tmp_path / 'uploads').exists()
    assert post_upload(c)[1]['state'] == 'committed'
    assert len(list((tmp_path / 'uploads' / 's1').iterdir())) == 1


def test_duplicate_route_uncertain_save_cannot_retire_live_writer(tmp_path):
    c = coordinator(make_store(tmp_path))
    assert c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)['writer']
    with fail_next_resource_directory_sync():
        assert post_upload(c)[0] == 503
    assert c.upload_status('s1', 'upload-1')['state'] == 'pending'
    assert not c.begin_upload('s1', 'upload-1', filename='hello.txt', size=5)['writer']
    assert finish(c)['state'] == 'committed'


def test_begin_route_retirement_save_failure_does_not_strand_reservation(tmp_path):
    store = make_store(tmp_path)
    c = coordinator(store)
    real_save = store.save_staged_attachments
    calls = 0
    def save():
        nonlocal calls
        calls += 1
        if calls == 1:
            with fail_next_resource_directory_sync():
                return real_save()
        if calls == 2:
            raise OSError('retirement save failed before replacement')
        return real_save()
    with patch.object(store, 'save_staged_attachments', side_effect=save):
        assert post_upload(c)[0] == 503
        assert c.upload_status('s1', 'upload-1')['state'] == 'failed'
        # The disk may still hold pending; a new process must also recover it.
        restored = make_store(tmp_path)
        restored.load_persistent_state()
        assert coordinator(restored).upload_status('s1', 'upload-1')['state'] == 'failed'
        assert post_upload(c)[1]['state'] == 'committed'
    assert len(list((tmp_path / 'uploads' / 's1').iterdir())) == 1
