from __future__ import annotations

import socketserver
from pathlib import Path

from codoxear.server_main import ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def test_http_server_remains_threaded_for_persistent_sse_handlers() -> None:
    assert issubclass(ThreadingHTTPServer, socketserver.ThreadingMixIn)
    assert ThreadingHTTPServer.daemon_threads is True


def test_eventsource_is_primary_and_message_polling_remains_fallback() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "new EventSource(url)" in source
    assert "`/api/sessions/${sessionId}/live?cursor=${encodeURIComponent(snapshot.liveCursor)}`" in source
    assert 'source.addEventListener("message"' in source
    assert 'source.addEventListener("error"' in source
    assert "kickPoll(0);" in source
    assert "scheduleMessageEventSourceRetry(sessionId, gen);" in source
    assert "if (appDisposed || !selected || messageSseOpen) return;" in source
    assert "if (appDisposed || messageSseOpen) return;" in source
    assert "`/api/sessions/${sid}/messages/live?cursor=${encodeURIComponent(reqCursor)}`" in source
