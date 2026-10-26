"""Coordinator recovery after snapshot identity publication outruns transfer."""
from pathlib import Path
from unittest.mock import patch

import pytest

from codoxear.file_upload import stage_uploaded_file
from test_attachment_resource import make_store, fail_next_resource_directory_sync
from test_session_snapshot_runtime import _Harness


def real_stores(h, root, *, load=False):
    store = make_store(root)
    if load:
        store.load_persistent_state()
    h.session_store = store
    h.coordinator.session_store = lambda: store
    h.coordinator.save_queues = lambda: store.save_queues(store.queues)
    h.coordinator.save_drafts = lambda: store.save_drafts(store.drafts)
    h.coordinator.save_unattended = lambda: store.save_unattended(store.unattended)
    h.coordinator.save_aliases = lambda: store.save_aliases(store.aliases)
    h.coordinator.save_sidebar_meta = lambda: store.save_sidebar_meta(store.sidebar_meta)
    return store


def save_other_stores(h):
    h.coordinator.save_queues()
    h.coordinator.save_drafts()
    h.coordinator.save_unattended()
    h.coordinator.save_aliases()
    h.coordinator.save_sidebar_meta()


@pytest.mark.parametrize('failure', ['before_attachment_publish', 'after_attachment_publish', 'queue_save'])
@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('automatic', [False, True])
def test_snapshot_intent_replays_all_stores_after_failed_remap(tmp_path, failure, restart, automatic):
    h = _Harness(tmp_path)
    store = real_stores(h, tmp_path / 'state')
    old, _ = h.add_live_session('broker-10', broker_pid=10, thread_id='t1')
    h.coordinator.save_now()
    path = stage_uploaded_file('broker-10', 'hello.txt', b'hello', upload_dir=store.paths.uploads_root,
                               now_fn=lambda: 100, max_bytes=512)
    store.add_staged_attachment('broker-10', {
        'id': 'a1', 'display_name': 'hello.txt', 'filename': path.name,
        'path': str(path), 'size': 5, 'created_ts': 100,
    })
    store.attachment_resource('broker-10')['uploads']['upload-1'] = {
        'state': 'committed', 'filename': 'hello.txt', 'size': 5, 'attachment_id': 'a1', 'digest': 'test-digest',
    }
    store.save_staged_attachments()
    store.queues = {'broker-10': [{'id': 'q1', 'text': 'first', 'created_ts': 1}],
                    'broker-30': [{'id': 'q2', 'text': 'second', 'created_ts': 2}]}
    store.drafts = {'broker-10': {'text': 'draft', 'updated_ts': 3}}
    store.aliases = {'broker-10': 'alias'}
    store.sidebar_meta = {'other': {'dependency_session_id': 'broker-10'}}
    store.unattended = {'thread:t1': {'enabled': True, 'request': 'continue', 'cooldown_minutes': 5, 'remaining_injections': 2}}
    save_other_stores(h)
    h.alive.discard(10)
    new, replacement = h.add_live_session('broker-30', broker_pid=30, thread_id='t1', start_ts=2)
    if failure == 'before_attachment_publish':
        fault = patch('codoxear.session_store.durable_write_resource', side_effect=OSError('no space'))
    elif failure == 'after_attachment_publish':
        fault = fail_next_resource_directory_sync()
    else:
        fault = patch.object(h.coordinator, 'save_queues', side_effect=OSError('queue save failed'))
    with fault:
        if automatic:
            h.coordinator.observe_registrations([replacement])
            h.flush()
            assert 'broker-30' in h.coordinator._dirty
        else:
            with pytest.raises(OSError, match='transfer incomplete'):
                h.coordinator.save_now()
    state = h.store.read()
    assert state.entries[('pi', 't1')].session_id == 'broker-30'
    assert len(state.remap_intents) == 1
    published_revision = store.attachment_resource('broker-30')['revision']
    if restart:
        h = _Harness(tmp_path)
        store = real_stores(h, tmp_path / 'state', load=True)
        new, replacement = h.add_live_session('broker-30', broker_pid=30, thread_id='t1', start_ts=2)
    # No supersede event can regenerate this remap: only durable intent can.
    assert h.store.upsert([h.coordinator.entry_for_session(new)]).remaps == []
    if automatic:
        if restart:
            h.coordinator.observe_registrations([replacement])
        h.flush()
    else:
        h.coordinator.save_now()
    assert h.store.read().remap_intents == ()
    assert [entry['id'] for entry in store.queues['broker-30']] == ['q1', 'q2']
    assert 'broker-10' not in store.queues
    assert store.drafts == {'broker-30': {'text': 'draft', 'updated_ts': 3}}
    assert store.aliases == {'broker-30': 'alias'}
    assert store.sidebar_meta['other']['dependency_session_id'] == 'broker-30'
    assert store.unattended['thread:t1']['request'] == 'continue'
    attachments = store.staged_attachments_for_session('broker-30')
    assert len(attachments) == 1 and attachments[0]['id'] == 'a1'
    assert Path(attachments[0]['path']).read_bytes() == b'hello'
    assert len(list((store.paths.uploads_root / 'broker-30').iterdir())) == 1
    if failure != 'before_attachment_publish':
        assert store.attachment_resource('broker-30')['revision'] == published_revision
    restored = make_store(tmp_path / 'state')
    restored.load_persistent_state()
    assert restored.queues == store.queues
    assert restored.drafts == store.drafts
    assert restored.aliases == store.aliases
