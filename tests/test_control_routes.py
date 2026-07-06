import base64
import json
from pathlib import Path

from codoxear.control_routes import ControlRouteDeps
from codoxear.control_routes import handle_control_post_route
from codoxear.server import _match_session_route


class NotReady(Exception):
    pass


class InjectionError(Exception):
    pass


class CommitUnknown(Exception):
    pass


class Handler:
    def __init__(self, body=None):
        self.body = body or {}
        self.unauthorized = False
        self.read_body_count = 0

    def _unauthorized(self):
        self.unauthorized = True


class Manager:
    def __init__(self):
        self.calls = []
        self.ready = True

    def delete_session(self, session_id):
        self.calls.append(("delete", session_id))
        return True

    def edit_session(self, session_id, **kwargs):
        self.calls.append(("edit", session_id, kwargs))
        return "alias", {"priority_offset": 0.25, "snooze_until": None, "dependency_session_id": None}

    def alias_set(self, session_id, name):
        self.calls.append(("rename", session_id, name))
        return name.strip()

    def clear_pending_attachment(self, session_id):
        self.calls.append(("clear_pending", session_id))
        return {"ok": True, "pending_attachment": False}

    def clear_commit_unknown_send(self, session_id):
        self.calls.append(("clear_unknown", session_id))
        return {"ok": True, "commit_unknown_send": False}

    def send(self, session_id, text, *, allow_pending_attachment=False):
        self.calls.append(("send", session_id, text, allow_pending_attachment))
        return {"queued": False, "queue_len": 0}

    def unattended_set(self, session_id, **kwargs):
        self.calls.append(("unattended", session_id, kwargs))
        return {"enabled": bool(kwargs.get("enabled")), "request": kwargs.get("request") or ""}

    def inject_keys(self, session_id, seq, *, interrupt=False):
        self.calls.append(("inject_keys", session_id, seq, interrupt))
        return {"ok": True}

    def attachment_injection_ready(self, session_id):
        self.calls.append(("ready", session_id))
        return self.ready

    def inject_attachment_keys(self, session_id, seq):
        self.calls.append(("inject_attachment", session_id, seq))
        return {"ok": True}


def _deps(responses, *, staged=None, auth=True):
    staged_paths = staged if staged is not None else []

    def stage(session_id, filename, raw):
        path = Path("/tmp") / session_id / filename
        staged_paths.append((path, raw))
        return path

    return ControlRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=lambda _handler, status, obj: responses.append((status, obj)),
        read_body=lambda handler: setattr(handler, "read_body_count", handler.read_body_count + 1) or b"",
        read_json_body=lambda handler, **_kwargs: handler.body,
        attach_upload_body_max_bytes=1024,
        attach_upload_max_bytes=512,
        stage_uploaded_file=stage,
        attachment_inject_text=lambda idx, path: f"Attachment {idx}: {path}\n" if idx > 0 else (_ for _ in ()).throw(ValueError("attachment_index must be >= 1")),
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, **_kwargs: int(value),
        session_not_ready_error=NotReady,
        session_injection_error=InjectionError,
        session_commit_unknown_error=CommitUnknown,
    )


def test_delete_route_reads_body_and_maps_unknown_to_404() -> None:
    class MissingManager(Manager):
        def delete_session(self, session_id):
            self.calls.append(("delete", session_id))
            return False

    responses = []
    handler = Handler()
    manager = MissingManager()
    assert handle_control_post_route(
        handler,
        path="/api/sessions/s1/delete",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert handler.read_body_count == 1
    assert manager.calls == [("delete", "s1")]
    assert responses == [(404, {"error": "unknown session"})]


def test_send_route_preserves_allow_pending_and_commit_unknown_status() -> None:
    class UnknownManager(Manager):
        def send(self, session_id, text, *, allow_pending_attachment=False):
            self.calls.append(("send", session_id, text, allow_pending_attachment))
            raise CommitUnknown("maybe sent")

    responses = []
    manager = UnknownManager()
    assert handle_control_post_route(
        Handler({"text": "go", "allow_pending_attachment": True}),
        path="/api/sessions/s1/send",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert manager.calls == [("send", "s1", "go", True)]
    assert responses == [(504, {"error": "maybe sent", "commit_unknown": True})]


def test_unattended_route_rejects_legacy_text_field_and_validates_numbers() -> None:
    responses = []
    manager = Manager()
    assert handle_control_post_route(
        Handler({"text": "old"}),
        path="/api/sessions/s1/unattended",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert responses == [(400, {"error": "unknown field: text (use request)"})]
    responses.clear()
    assert handle_control_post_route(
        Handler({"enabled": True, "request": "continue", "cooldown_minutes": "3", "remaining_injections": "2"}),
        path="/api/sessions/s1/unattended",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert responses == [(200, {"ok": True, "enabled": True, "request": "continue"})]
    assert manager.calls == [("unattended", "s1", {"enabled": True, "request": "continue", "cooldown_minutes": 3, "remaining_injections": 2})]


def test_interrupt_route_sends_escaped_esc_sequence() -> None:
    responses = []
    manager = Manager()
    handler = Handler()
    assert handle_control_post_route(
        handler,
        path="/api/sessions/s1/interrupt",
        manager=manager,
        deps=_deps(responses),
        match_session_route=_match_session_route,
    ) is True
    assert handler.read_body_count == 1
    assert manager.calls == [("inject_keys", "s1", "\\x1b", True)]
    assert responses == [(200, {"ok": True, "broker": {"ok": True}})]


def test_inject_attachment_checks_readiness_before_decoding_or_staging() -> None:
    responses = []
    staged = []
    manager = Manager()
    manager.ready = False
    assert handle_control_post_route(
        Handler({"filename": "note.txt", "attachment_index": 1, "data_b64": base64.b64encode(b"hello").decode("ascii")}),
        path="/api/sessions/s1/inject_file",
        manager=manager,
        deps=_deps(responses, staged=staged),
        match_session_route=_match_session_route,
    ) is True
    assert manager.calls == [("ready", "s1")]
    assert staged == []
    assert responses == [(409, {"error": "session is busy; wait before attaching a file"})]


def test_inject_attachment_stages_and_injects_bracketed_paste() -> None:
    responses = []
    staged = []
    manager = Manager()
    assert handle_control_post_route(
        Handler({"filename": "note.txt", "attachment_index": 1, "data_b64": base64.b64encode(b"hello").decode("ascii")}),
        path="/api/sessions/s1/inject_image",
        manager=manager,
        deps=_deps(responses, staged=staged),
        match_session_route=_match_session_route,
    ) is True
    path = Path("/tmp") / "s1" / "note.txt"
    assert staged == [(path, b"hello")]
    assert manager.calls[0] == ("ready", "s1")
    assert manager.calls[1][0:2] == ("inject_attachment", "s1")
    assert "\x1b[200~Attachment 1: /tmp/s1/note.txt\n\x1b[201~" == manager.calls[1][2]
    assert responses == [(200, {"ok": True, "path": str(path), "inject_text": f"Attachment 1: {path}\n", "broker": {"ok": True}})]


# ---------------------------------------------------------------------------
# Full status matrix for the inject-attachment route (`_handle_inject_attachment`),
# driven entirely through the injected-deps seam (`handle_control_post_route` +
# `ControlRouteDeps`). This replaces the former `do_POST`/server-global
# monkeypatch coverage in `tests/test_file_upload.py`.
# ---------------------------------------------------------------------------

def _default_stage(session_id, filename, raw):
    return Path("/tmp") / session_id / filename


def _default_inject_text(idx, path):
    if idx <= 0:
        raise ValueError("attachment_index must be >= 1")
    return f"Attachment {idx}: {path}\n"


def _inject_deps(
    responses,
    *,
    auth=True,
    stage=_default_stage,
    inject_text=_default_inject_text,
    body_max=1024,
    upload_max=512,
):
    """Build ControlRouteDeps for inject-attachment matrix tests.

    `stage`/`inject_text` are overridable so each matrix case can force the
    exact failure mode without touching server globals."""
    return ControlRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=lambda _handler, status, obj: responses.append((status, obj)),
        read_body=lambda handler: setattr(handler, "read_body_count", handler.read_body_count + 1) or b"",
        read_json_body=lambda handler, **_kwargs: handler.body,
        attach_upload_body_max_bytes=body_max,
        attach_upload_max_bytes=upload_max,
        stage_uploaded_file=stage,
        attachment_inject_text=inject_text,
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, **_kwargs: int(value),
        session_not_ready_error=NotReady,
        session_injection_error=InjectionError,
        session_commit_unknown_error=CommitUnknown,
    )


def _good_body(**overrides):
    body = {"filename": "note.txt", "attachment_index": 1, "data_b64": base64.b64encode(b"hello").decode("ascii")}
    body.update(overrides)
    return body


def _post(handler_body, responses, *, manager=None, auth=True, stage=_default_stage, inject_text=_default_inject_text):
    return handle_control_post_route(
        Handler(handler_body),
        path="/api/sessions/s1/inject_file",
        manager=manager or Manager(),
        deps=_inject_deps(responses, auth=auth, stage=stage, inject_text=inject_text),
        match_session_route=_match_session_route,
    )


def test_inject_attachment_happy_path_body_is_json_serializable_and_bracketed() -> None:
    responses = []
    manager = Manager()
    assert _post(_good_body(), responses, manager=manager) is True
    status, obj = responses[-1]
    assert status == 200
    # The whole 200 body must survive real json serialization (this recovers
    # the coverage the former _json_response-patching test blinded).
    json.dumps(obj)
    assert obj["ok"] is True
    assert obj["inject_text"].endswith("\n")
    assert obj["inject_text"].startswith("Attachment 1: ")
    assert obj["path"] == str(Path("/tmp") / "s1" / "note.txt")
    # The injected PTY sequence is wrapped in bracketed-paste markers.
    seq = manager.calls[1][2]
    assert seq.startswith("\x1b[200~") and seq.endswith("\x1b[201~")


def test_inject_attachment_rejects_unauthorized_handler() -> None:
    responses = []
    handler = Handler(_good_body())
    # require_auth returns False -> handler._unauthorized() invoked, no response emitted.
    assert handle_control_post_route(
        handler,
        path="/api/sessions/s1/inject_file",
        manager=Manager(),
        deps=_inject_deps(responses, auth=False),
        match_session_route=_match_session_route,
    ) is True
    assert handler.unauthorized is True
    assert responses == []


def test_inject_attachment_missing_filename_is_400() -> None:
    responses = []
    assert _post(_good_body(filename="   "), responses) is True
    assert responses == [(400, {"error": "filename required"})]
    responses.clear()
    assert _post({"attachment_index": 1, "data_b64": "AAA"}, responses) is True
    assert responses == [(400, {"error": "filename required"})]


def test_inject_attachment_non_integer_index_is_400() -> None:
    responses = []
    # bool is rejected even though bool is a subclass of int.
    assert _post(_good_body(attachment_index=True), responses) is True
    assert responses == [(400, {"error": "attachment_index must be an integer"})]
    responses.clear()
    assert _post(_good_body(attachment_index="1"), responses) is True
    assert responses == [(400, {"error": "attachment_index must be an integer"})]


def test_inject_attachment_missing_data_b64_is_400() -> None:
    responses = []
    assert _post(_good_body(data_b64=""), responses) is True
    assert responses == [(400, {"error": "data_b64 required"})]
    responses.clear()
    assert _post({"filename": "note.txt", "attachment_index": 1}, responses) is True
    assert responses == [(400, {"error": "data_b64 required"})]


def test_inject_attachment_invalid_base64_is_400() -> None:
    responses = []
    assert _post(_good_body(data_b64="!!!not-base64!!!"), responses) is True
    assert responses == [(400, {"error": "invalid base64"})]


def test_inject_attachment_stage_oversize_is_413() -> None:
    responses = []
    def _oversize(session_id, filename, raw):
        raise ValueError("file too large (max 512 bytes)")
    assert _post(_good_body(), responses, stage=_oversize) is True
    assert responses == [(413, {"error": "file too large (max 512 bytes)"})]


def test_inject_attachment_stage_generic_value_error_is_400() -> None:
    responses = []
    def _bad_path(session_id, filename, raw):
        raise ValueError("bad path")
    assert _post(_good_body(), responses, stage=_bad_path) is True
    assert responses == [(400, {"error": "bad path"})]


def test_inject_attachment_inject_text_value_error_is_400() -> None:
    responses = []
    def _bad_index(idx, path):
        raise ValueError("attachment_index must be >= 1")
    assert _post(_good_body(attachment_index=0), responses, inject_text=_bad_index) is True
    assert responses == [(400, {"error": "attachment_index must be >= 1"})]


def test_inject_attachment_readiness_key_error_is_404() -> None:
    responses = []
    class UnknownSession(Manager):
        def attachment_injection_ready(self, session_id):
            raise KeyError(session_id)
    assert _post(_good_body(), responses, manager=UnknownSession()) is True
    assert responses == [(404, {"error": "unknown session"})]


def test_inject_attachment_readiness_not_ready_is_409() -> None:
    responses = []
    class BusySession(Manager):
        def attachment_injection_ready(self, session_id):
            raise NotReady("session is busy")
    assert _post(_good_body(), responses, manager=BusySession()) is True
    assert responses == [(409, {"error": "session is busy"})]


def test_inject_attachment_readiness_generic_exception_is_409() -> None:
    responses = []
    class GlitchySession(Manager):
        def attachment_injection_ready(self, session_id):
            raise RuntimeError("state unavailable")
    assert _post(_good_body(), responses, manager=GlitchySession()) is True
    assert responses == [(409, {"error": "session state unavailable; wait before attaching a file"})]


def test_inject_attachment_readiness_false_is_409() -> None:
    responses = []
    manager = Manager()
    manager.ready = False
    assert _post(_good_body(), responses, manager=manager) is True
    assert responses == [(409, {"error": "session is busy; wait before attaching a file"})]


def test_inject_attachment_inject_key_error_is_404() -> None:
    responses = []
    class VanishedSession(Manager):
        def inject_attachment_keys(self, session_id, seq):
            raise KeyError(session_id)
    assert _post(_good_body(), responses, manager=VanishedSession()) is True
    assert responses == [(404, {"error": "unknown session"})]


def test_inject_attachment_injection_error_is_502() -> None:
    responses = []
    class BrokenBroker(Manager):
        def inject_attachment_keys(self, session_id, seq):
            raise InjectionError("broker write failed")
    assert _post(_good_body(), responses, manager=BrokenBroker()) is True
    assert responses == [(502, {"error": "broker write failed"})]


def test_inject_attachment_commit_unknown_is_504_with_flag() -> None:
    responses = []
    class UncertainBroker(Manager):
        def inject_attachment_keys(self, session_id, seq):
            raise CommitUnknown("commit status unknown")
    assert _post(_good_body(), responses, manager=UncertainBroker()) is True
    assert responses == [(504, {"error": "commit status unknown", "commit_unknown": True})]


def test_inject_attachment_ready_check_precedes_base64_decode() -> None:
    # A not-ready session must short-circuit before base64 decoding, so an
    # invalid payload never reaches the decoder when the session is busy.
    responses = []
    manager = Manager()
    manager.ready = False
    assert _post(_good_body(data_b64="!!!invalid!!!"), responses, manager=manager) is True
    assert responses == [(409, {"error": "session is busy; wait before attaching a file"})]
