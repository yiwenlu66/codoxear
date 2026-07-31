import threading
from pathlib import Path

from codoxear import server


def setup_function() -> None:
    with server._FILE_WRITE_LOCKS_LOCK:
        server._FILE_WRITE_LOCKS.clear()


def teardown_function() -> None:
    with server._FILE_WRITE_LOCKS_LOCK:
        server._FILE_WRITE_LOCKS.clear()


def test_file_write_lock_entry_is_removed_after_use(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"

    with server._file_write_lock(path):
        with server._FILE_WRITE_LOCKS_LOCK:
            assert str(path) in server._FILE_WRITE_LOCKS
            assert server._FILE_WRITE_LOCKS[str(path)][1] == 1

    with server._FILE_WRITE_LOCKS_LOCK:
        assert str(path) not in server._FILE_WRITE_LOCKS


def test_file_write_lock_counts_waiters_before_acquire(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    waiter_started = threading.Event()
    release_holder = threading.Event()
    waiter_done = threading.Event()

    def waiter() -> None:
        with server._file_write_lock(path):
            waiter_started.set()
            release_holder.wait(timeout=5)
        waiter_done.set()

    try:
        with server._file_write_lock(path):
            thread = threading.Thread(target=waiter)
            thread.start()
            # Wait until the waiter has registered its refcount while blocked on the per-file lock.
            for _ in range(200):
                with server._FILE_WRITE_LOCKS_LOCK:
                    entry = server._FILE_WRITE_LOCKS.get(str(path))
                    if entry and entry[1] == 2:
                        break
                waiter_started.wait(0.01)
            with server._FILE_WRITE_LOCKS_LOCK:
                assert server._FILE_WRITE_LOCKS[str(path)][1] == 2
    finally:
        release_holder.set()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert waiter_done.is_set()
    with server._FILE_WRITE_LOCKS_LOCK:
        assert str(path) not in server._FILE_WRITE_LOCKS
