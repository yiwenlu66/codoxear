from codoxear.queue_routes import QueueRouteDeps
from codoxear.queue_routes import handle_queue_get_route
from codoxear.queue_routes import handle_queue_post_route
from codoxear.server import _match_session_route


class NotReady(Exception):
    pass


class Handler:
    def __init__(self, body=None):
        self.body = body or {}
        self.unauthorized = False

    def _unauthorized(self):
        self.unauthorized = True


class Manager:
    def __init__(self):
        self.calls = []
        self.items = [{"id": "q1", "text": "hello"}]

    def queue_list(self, session_id):
        self.calls.append(("list", session_id))
        return list(self.items)

    def enqueue(self, session_id, text):
        self.calls.append(("enqueue", session_id, text))
        return {"queued": True, "queue_len": 1}

    def queue_delete(self, session_id, item_id, *, allow_commit_unknown=False, allow_orphan_recovery=False):
        self.calls.append(("delete", session_id, item_id, allow_commit_unknown, allow_orphan_recovery))
        return {"ok": True, "queue_len": 0}

    def queue_update(self, session_id, item_id, text):
        self.calls.append(("update", session_id, item_id, text))
        return {"ok": True, "queue_len": 1}

    def queue_move(self, session_id, item_id, to_index):
        self.calls.append(("move", session_id, item_id, to_index))
        return {"ok": True, "queue_len": 1}


def _deps(responses, *, auth=True):
    return QueueRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=lambda _handler, status, obj: responses.append((status, obj)),
        read_json_body=lambda handler: handler.body,
        session_not_ready_error=NotReady,
    )


def test_queue_get_returns_modern_items_and_legacy_text_queue() -> None:
    responses = []
    manager = Manager()
    handled = handle_queue_get_route(
        Handler(),
        path="/api/sessions/s1/queue",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [(200, {"ok": True, "items": [{"id": "q1", "text": "hello"}], "queue": ["hello"]})]
    assert manager.calls == [("list", "s1")]


def test_queue_get_auth_failure_stops_before_manager() -> None:
    responses = []
    manager = Manager()
    handler = Handler()
    handled = handle_queue_get_route(
        handler,
        path="/api/sessions/s1/queue",
        manager=manager,
        deps=_deps(responses, auth=False),
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert handler.unauthorized is True
    assert responses == []
    assert manager.calls == []


def test_enqueue_maps_not_ready_to_conflict() -> None:
    class BusyManager(Manager):
        def enqueue(self, session_id, text):
            raise NotReady("session is busy")

    responses = []
    handled = handle_queue_post_route(
        Handler({"text": "later"}),
        path="/api/sessions/s1/enqueue",
        manager=BusyManager(),
        deps=_deps(responses),
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [(409, {"error": "session is busy"})]


def test_queue_delete_validates_confirmation_flags_and_calls_manager() -> None:
    responses = []
    manager = Manager()
    handled = handle_queue_post_route(
        Handler({"id": "q1", "allow_commit_unknown": True, "allow_orphan_recovery": True}),
        path="/api/sessions/s1/queue/delete",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [(200, {"ok": True, "queue_len": 0})]
    assert manager.calls == [("delete", "s1", "q1", True, True)]


def test_queue_delete_rejects_non_boolean_confirmation_flags() -> None:
    responses = []
    handled = handle_queue_post_route(
        Handler({"id": "q1", "allow_commit_unknown": "yes"}),
        path="/api/sessions/s1/queue/delete",
        manager=Manager(),
        deps=_deps(responses),
        match_session_route=_match_session_route,
    )
    assert handled is True
    assert responses == [(400, {"error": "allow_commit_unknown must be a boolean"})]


def test_queue_update_and_move_validate_payloads() -> None:
    responses = []
    manager = Manager()
    assert handle_queue_post_route(
        Handler({"id": "q1", "text": "edited"}),
        path="/api/sessions/s1/queue/update",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert handle_queue_post_route(
        Handler({"id": "q1", "to_index": True}),
        path="/api/sessions/s1/queue/move",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert responses == [(200, {"ok": True, "queue_len": 1}), (400, {"error": "to_index required"})]
    assert manager.calls == [("update", "s1", "q1", "edited")]
