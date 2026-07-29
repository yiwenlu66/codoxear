from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

from codoxear.voice_routes import VoiceRouteDeps
from codoxear.voice_routes import handle_voice_get_route
from codoxear.voice_routes import handle_voice_post_route


class _FakeHandler:
    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: list[tuple[str, str]] = []
        self.errors: list[int] = []
        self.unauthorized = False
        self.wfile = BytesIO()

    def _unauthorized(self) -> None:
        self.unauthorized = True

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        return None

    def send_error(self, status: int) -> None:
        self.errors.append(status)


class _FakeVoicePush:
    def __init__(self, *, segment_path: Path | None = None) -> None:
        self.segment_path_value = segment_path
        self.settings_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.toggle_calls: list[dict[str, object]] = []
        self.listener_calls: list[dict[str, object]] = []
        self.raise_toggle_keyerror = False

    def settings_snapshot(self, *, redact_secrets: bool = False) -> dict[str, object]:
        return {"redact_secrets": redact_secrets, "tts_api_key": "", "has_tts_api_key": True}

    def subscriptions_snapshot(self) -> dict[str, object]:
        return {"subscriptions": [{"id": "sub-1"}]}

    def notification_state_for_message(self, message_id: str):
        if message_id == "known":
            return {"message_id": message_id, "summary_status": "done"}
        return None

    def notification_feed_since(self, since_ts: float):
        return [{"since": since_ts}]

    def playlist_bytes(self) -> bytes:
        return b"#EXTM3U\n"

    def segment_path(self, segment_name: str) -> Path:
        if self.segment_path_value is None:
            raise FileNotFoundError(segment_name)
        return self.segment_path_value

    def set_settings(self, obj, *, preserve_blank_api_key: bool, redact_response: bool):
        self.settings_calls.append(
            {"obj": obj, "preserve_blank_api_key": preserve_blank_api_key, "redact_response": redact_response}
        )
        return {"saved": True}

    def upsert_subscription(self, *, subscription, user_agent: str, device_label: str, device_class: str):
        self.upsert_calls.append(
            {
                "subscription": subscription,
                "user_agent": user_agent,
                "device_label": device_label,
                "device_class": device_class,
            }
        )
        return {"subscriptions": [{"id": "sub-1"}]}

    def toggle_subscription(self, *, endpoint: str, enabled: bool):
        self.toggle_calls.append({"endpoint": endpoint, "enabled": enabled})
        if self.raise_toggle_keyerror:
            raise KeyError(endpoint)
        return {"subscriptions": [{"endpoint": endpoint, "notifications_enabled": enabled}]}

    def listener_heartbeat(self, *, client_id: str, enabled: bool):
        self.listener_calls.append({"client_id": client_id, "enabled": enabled})
        return {"active_listener_count": 1 if enabled else 0}


def _deps(*, body: dict[str, object] | None = None, auth: bool = True, prompt: str = "Default prompt"):
    responses: list[tuple[int, dict[str, object]]] = []
    prompt_state = {"value": prompt}

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    def save_unattended_prompt(value: str) -> str:
        prompt_state["value"] = value
        return value

    deps = VoiceRouteDeps(
        require_auth=lambda _handler: auth,
        json_response=json_response,
        read_json_body=lambda _handler, **_kwargs: dict(body or {}),
        load_unattended_prompt=lambda: prompt_state["value"],
        save_unattended_prompt=save_unattended_prompt,
        default_unattended_prompt="Default prompt",
    )
    return deps, responses


def test_voice_get_settings_feed_and_message_status_mapping() -> None:
    voice = _FakeVoicePush()
    deps, responses = _deps()
    handler = _FakeHandler()

    assert handle_voice_get_route(handler, path="/api/settings/voice", query="", voice_push=voice, deps=deps) is True
    assert handle_voice_get_route(
        handler, path="/api/notifications/message", query="message_id=known", voice_push=voice, deps=deps
    ) is True
    assert handle_voice_get_route(
        handler, path="/api/notifications/message", query="message_id=", voice_push=voice, deps=deps
    ) is True
    assert handle_voice_get_route(
        handler, path="/api/notifications/feed", query="since=bad", voice_push=voice, deps=deps
    ) is True
    assert responses == [
        (200, {"ok": True, "redact_secrets": True, "tts_api_key": "", "has_tts_api_key": True}),
        (200, {"ok": True, "message_id": "known", "summary_status": "done"}),
        (400, {"error": "message_id required"}),
        (400, {"error": "invalid since"}),
    ]


def test_unattended_prompt_get_post_and_validation_mapping() -> None:
    voice = _FakeVoicePush()
    handler = _FakeHandler()
    deps, responses = _deps(prompt="Custom constitution")

    assert handle_voice_get_route(handler, path="/api/settings/unattended-prompt", query="", voice_push=voice, deps=deps) is True
    assert responses == [(200, {"ok": True, "prompt": "Custom constitution", "default_prompt": "Default prompt"})]

    deps, responses = _deps(body={"prompt": "Updated constitution"})
    assert handle_voice_post_route(handler, path="/api/settings/unattended-prompt", voice_push=voice, deps=deps) is True
    assert responses == [(200, {"ok": True, "prompt": "Updated constitution", "default_prompt": "Default prompt"})]

    deps, responses = _deps(body={"prompt": 7})
    assert handle_voice_post_route(handler, path="/api/settings/unattended-prompt", voice_push=voice, deps=deps) is True
    assert responses == [(400, {"error": "prompt must be a string"})]


def test_voice_get_audio_playlist_and_segments_use_no_store_headers() -> None:
    with tempfile.TemporaryDirectory() as td:
        segment = Path(td) / "seg.ts"
        segment.write_bytes(b"segment")
        deps, responses = _deps()
        playlist_handler = _FakeHandler()
        assert handle_voice_get_route(
            playlist_handler,
            path="/api/audio/live.m3u8",
            query="",
            voice_push=_FakeVoicePush(segment_path=segment),
            deps=deps,
        ) is True
        assert responses == []
        assert playlist_handler.status == 200
        assert playlist_handler.wfile.getvalue() == b"#EXTM3U\n"
        assert ("Content-Type", "application/vnd.apple.mpegurl") in playlist_handler.headers
        assert ("Cache-Control", "no-store") in playlist_handler.headers
        assert ("Pragma", "no-cache") in playlist_handler.headers
        assert ("Expires", "0") in playlist_handler.headers

        segment_handler = _FakeHandler()
        assert handle_voice_get_route(
            segment_handler,
            path="/api/audio/segments/seg.ts",
            query="",
            voice_push=_FakeVoicePush(segment_path=segment),
            deps=deps,
        ) is True
        assert segment_handler.status == 200
        assert segment_handler.wfile.getvalue() == b"segment"
        assert ("Content-Type", "video/mp2t") in segment_handler.headers

        missing_handler = _FakeHandler()
        assert handle_voice_get_route(
            missing_handler,
            path="/api/audio/segments/missing.ts",
            query="",
            voice_push=_FakeVoicePush(),
            deps=deps,
        ) is True
        assert missing_handler.errors == [404]


def test_voice_post_settings_subscription_toggle_and_listener_mapping() -> None:
    voice = _FakeVoicePush()
    handler = _FakeHandler()
    deps, responses = _deps(body={"tts_api_key": "", "subscription": {"endpoint": "e"}, "user_agent": "ua"})
    assert handle_voice_post_route(handler, path="/api/settings/voice", voice_push=voice, deps=deps) is True
    assert voice.settings_calls == [
        {"obj": {"tts_api_key": "", "subscription": {"endpoint": "e"}, "user_agent": "ua"}, "preserve_blank_api_key": True, "redact_response": True}
    ]
    assert responses == [(200, {"ok": True, "saved": True})]

    deps, responses = _deps(body={"subscription": {"endpoint": "e"}, "user_agent": "ua", "device_label": "phone", "device_class": "mobile"})
    assert handle_voice_post_route(handler, path="/api/notifications/subscription", voice_push=voice, deps=deps) is True
    assert voice.upsert_calls[-1] == {
        "subscription": {"endpoint": "e"},
        "user_agent": "ua",
        "device_label": "phone",
        "device_class": "mobile",
    }
    assert responses == [(200, {"ok": True, "subscriptions": [{"id": "sub-1"}]})]

    deps, responses = _deps(body={"endpoint": "e", "enabled": True})
    assert handle_voice_post_route(handler, path="/api/notifications/subscription/toggle", voice_push=voice, deps=deps) is True
    assert responses == [(200, {"ok": True, "subscriptions": [{"endpoint": "e", "notifications_enabled": True}]})]

    deps, responses = _deps(body={"client_id": "listener-1", "enabled": True})
    assert handle_voice_post_route(handler, path="/api/audio/listener", voice_push=voice, deps=deps) is True
    assert voice.listener_calls[-1] == {"client_id": "listener-1", "enabled": True}
    assert responses == [(200, {"ok": True, "active_listener_count": 1})]


def test_voice_post_validation_and_unknown_subscription_errors() -> None:
    voice = _FakeVoicePush()
    handler = _FakeHandler()
    deps, responses = _deps(body={"endpoint": "", "enabled": True})
    assert handle_voice_post_route(handler, path="/api/notifications/subscription/toggle", voice_push=voice, deps=deps) is True
    assert responses == [(400, {"error": "endpoint required"})]

    deps, responses = _deps(body={"endpoint": "e", "enabled": "yes"})
    assert handle_voice_post_route(handler, path="/api/notifications/subscription/toggle", voice_push=voice, deps=deps) is True
    assert responses == [(400, {"error": "enabled must be a boolean"})]

    voice.raise_toggle_keyerror = True
    deps, responses = _deps(body={"endpoint": "e", "enabled": False})
    assert handle_voice_post_route(handler, path="/api/notifications/subscription/toggle", voice_push=voice, deps=deps) is True
    assert responses == [(404, {"error": "unknown subscription"})]

    deps, responses = _deps(body={"client_id": "", "enabled": True})
    assert handle_voice_post_route(handler, path="/api/audio/listener", voice_push=voice, deps=deps) is True
    assert responses == [(400, {"error": "client_id required"})]


def test_voice_routes_preserve_auth_failure_short_circuit() -> None:
    voice = _FakeVoicePush()
    handler = _FakeHandler()
    deps, responses = _deps(auth=False)
    assert handle_voice_get_route(handler, path="/api/settings/voice", query="", voice_push=voice, deps=deps) is True
    assert handler.unauthorized is True
    assert responses == []
