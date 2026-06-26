import threading
from pathlib import Path
from unittest.mock import patch

from codoxear import server
from codoxear.server import SessionManager


def test_session_manager_persistent_maps_are_store_owned() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()

    aliases = {"s1": "alpha"}
    mgr._aliases = aliases
    assert mgr._session_store_for_manager().aliases is aliases
    assert mgr._aliases is aliases

    queues = {"s1": [{"id": "q1", "text": "queued"}]}
    mgr._queues = queues
    assert mgr._session_store_for_manager().queues is queues
    assert mgr._queues is queues

    pending = {"s1"}
    mgr._pending_attachment_ids = pending
    assert mgr._session_store_for_manager().pending_attachment_ids is pending
    assert mgr._pending_attachment_ids is pending


def test_session_store_rebinds_paths_without_losing_in_memory_state() -> None:
    mgr = SessionManager.__new__(SessionManager)
    mgr._lock = threading.Lock()
    mgr._aliases = {"s1": "alpha"}
    original_store = mgr._session_store_for_manager()
    with patch.object(server, "ALIAS_PATH", Path("/tmp/codoxear-test-aliases.json")):
        rebound_store = mgr._session_store_for_manager()
    assert rebound_store is not original_store
    assert rebound_store.paths.aliases == Path("/tmp/codoxear-test-aliases.json")
    assert rebound_store.aliases == {"s1": "alpha"}
    assert mgr._aliases == {"s1": "alpha"}
