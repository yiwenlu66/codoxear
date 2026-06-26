from __future__ import annotations

from codoxear.hook_routes import HookRouteDeps
from codoxear.hook_routes import handle_hook_post_route


class _FakeHandler:
    pass


def test_hook_notify_drains_body_and_returns_ignored_response() -> None:
    handler = _FakeHandler()
    calls: list[object] = []
    responses: list[tuple[int, dict[str, object]]] = []

    deps = HookRouteDeps(
        read_body=lambda h: calls.append(h) or b"payload",
        json_response=lambda _h, status, payload: responses.append((status, payload)),
    )

    assert handle_hook_post_route(handler, path="/api/hooks/notify", deps=deps) is True
    assert calls == [handler]
    assert responses == [(200, {"ignored": True})]


def test_hook_route_ignores_other_paths_without_side_effects() -> None:
    deps = HookRouteDeps(
        read_body=lambda _h: (_ for _ in ()).throw(AssertionError("body should not be read")),
        json_response=lambda *_args: (_ for _ in ()).throw(AssertionError("response should not be written")),
    )

    assert handle_hook_post_route(_FakeHandler(), path="/api/other", deps=deps) is False


def test_hook_read_errors_propagate_to_route_exception_handler() -> None:
    class SentinelError(Exception):
        pass

    deps = HookRouteDeps(
        read_body=lambda _h: (_ for _ in ()).throw(SentinelError("bad body")),
        json_response=lambda *_args: None,
    )

    try:
        handle_hook_post_route(_FakeHandler(), path="/api/hooks/notify", deps=deps)
    except SentinelError:
        pass
    else:
        raise AssertionError("expected body read error to propagate")
