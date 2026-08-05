from __future__ import annotations

import json
from pathlib import Path

from codoxear.codex_live_control import update_codex_thread_settings


class MockAppServer:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self._responses = [
            {"id": 1, "result": {}},
            {"id": 2, "result": {}},
        ]
        self.closed = False

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def recv(self, *, timeout: float) -> str:
        assert timeout > 0
        return json.dumps(self._responses.pop(0))

    def close(self) -> None:
        self.closed = True


def test_effort_control_sends_typed_thread_update_without_provider() -> None:
    app_server = MockAppServer()

    result = update_codex_thread_settings(
        Path("/tmp/codex-live-control.sock"),
        thread_id="thread-for-effort",
        effort="low",
        connect=lambda _path, *, timeout_s: app_server,
    )

    assert result == {"ok": True, "model": None, "effort": "low"}
    request = app_server.sent[-1]
    assert request == {
        "method": "thread/settings/update",
        "id": 2,
        "params": {"threadId": "thread-for-effort", "effort": "low"},
    }
    assert "provider" not in request["params"]
    assert app_server.closed is True
