from __future__ import annotations
from frontend_module_loader import module_path

import json
import subprocess
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

from codoxear.rollout_delivery import _extract_delivery_messages
from codoxear.voice_push import VoicePushCoordinator
from codoxear.voice_routes import VoiceRouteDeps
from codoxear.voice_routes import handle_voice_get_route


ROOT = Path(__file__).resolve().parents[1]
VOICE_SOURCE = (module_path("app_voice.js")).read_text(encoding="utf-8")
VOICE_HELPERS_SOURCE = (module_path("app_voice_helpers.js")).read_text(encoding="utf-8")
NOTIFICATIONS_SOURCE = (module_path("app_notifications.js")).read_text(encoding="utf-8")
MODAL_SOURCE = (module_path("app_modal.js")).read_text(encoding="utf-8")


class _RouteHandler:
    pass


def _notification_feed_from_route(coordinator: VoicePushCoordinator) -> list[dict[str, Any]]:
    responses: list[tuple[int, dict[str, Any]]] = []
    deps = VoiceRouteDeps(
        require_auth=lambda _handler: True,
        json_response=lambda _handler, status, payload: responses.append((status, payload)),
        read_json_body=lambda _handler, **_kwargs: {},
        load_unattended_prompt=lambda: "",
        save_unattended_prompt=lambda value: value,
        default_unattended_prompt="",
    )

    assert handle_voice_get_route(
        _RouteHandler(),
        path="/api/notifications/feed",
        query="since=0",
        voice_push=coordinator,
        deps=deps,
    ) is True
    assert len(responses) == 1
    status, payload = responses[0]
    assert status == 200
    assert payload["ok"] is True
    items = payload["items"]
    assert isinstance(items, list)
    return items


def _delivery_after_refresh(feed_items: list[dict[str, Any]]) -> dict[str, Any]:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const feedItems = {json.dumps(feed_items)};
        const notifications = [];
        const storage = new Map([["codoxear.notificationEnabled", "1"], ["codoxear.desktopNotificationsEnabled", "1"]]);
        class NotificationCtor {{
          constructor(title, options) {{ notifications.push([title, options.body]); }}
          close() {{}}
        }}
        NotificationCtor.permission = "granted";
        class AudioContextCtor {{
          constructor() {{ this.state = "running"; this.currentTime = 0; this.destination = {{}}; }}
          resume() {{ return Promise.resolve(); }}
          createOscillator() {{ return {{ frequency: {{}}, connect() {{}}, start() {{}}, stop() {{}} }}; }}
          createGain() {{ return {{ gain: {{ setValueAtTime() {{}}, exponentialRampToValueAtTime() {{}} }}, connect() {{}} }}; }}
          close() {{ return Promise.resolve(); }}
        }}
        function node() {{
          return {{
            style: {{}}, dataset: {{}}, textContent: "", title: "", disabled: false, open: false, value: "", checked: false,
            _children: [], _attrs: {{}},
            classList: {{
              values: new Set(),
              toggle(name, enabled) {{ enabled ? this.values.add(name) : this.values.delete(name); }},
              add(name) {{ this.values.add(name); }}, remove(name) {{ this.values.delete(name); }},
              contains(name) {{ return this.values.has(name); }},
            }},
            setAttribute(name, value) {{ this._attrs[name] = String(value); }},
            removeAttribute(name) {{ delete this._attrs[name]; }},
            append(...children) {{ this._children.push(...children); }},
            appendChild(child) {{ this._children.push(child); return child; }},
            replaceChildren(...children) {{ this._children = children; }},
            addEventListener() {{}}, removeEventListener() {{}}, matches() {{ return false; }},
            load() {{}}, pause() {{}}, play() {{ return Promise.resolve(); }},
          }};
        }}
        const dom = {{
          announceBtn: node(), notificationBtn: node(), liveAudio: node(),
          voiceSettingsStatus: node(), voiceBaseUrlInput: node(), voiceApiKeyInput: node(), voiceClearApiKeyToggle: node(), narrationSettingToggle: node(),
          voiceSettingsCancelBtn: node(), voiceSettingsSaveBtn: node(),
        }};
        const documentTarget = {{ activeElement: null, contains: () => true, createElement: () => node() }};
        const ctx = {{
          HTMLElement: function HTMLElement() {{}}, console, Notification: NotificationCtor,
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11 Linux x86_64" }}, document: documentTarget,
        }};
        vm.createContext(ctx);
        for (const source of [{json.dumps(MODAL_SOURCE)}, {json.dumps(VOICE_HELPERS_SOURCE)}, {json.dumps(NOTIFICATIONS_SOURCE)}, {json.dumps(VOICE_SOURCE)}]) vm.runInContext(source, ctx);
        const controller = ctx.window.CodoxearVoice.createVoiceController({{
          ...dom, windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
          notificationOptions: {{
            voiceHost: {{ style: {{}}, appendChild() {{}}, insertBefore() {{}}, firstChild: null }},
            el: (_tag, attrs = {{}}, children = []) => {{ const created = attrs.id && dom[attrs.id] ? dom[attrs.id] : node(); created.append(...children); return created; }}, iconSvg: () => "",
            isAppDisposed: () => false, api: async (url, options = {{}}) => {{
              if (url.includes("/api/notifications/feed")) return {{ items: typeof feeds !== "undefined" ? (feeds.shift() || []) : typeof feedItems !== "undefined" ? feedItems.splice(0) : [] }};
              if (url.includes("/api/notifications/subscription")) return {{ subscriptions: [] }};
              return {{}};
            }}, setToast() {{}}, handleAppAuthLoss() {{}}, resolveAppUrl: (x) => x, versionedShellAssetPath: (x) => x,
            storageGetItem: (key) => storage.get(key) || null,
            storageSetItem: (key, value) => storage.set(key, String(value)), storageRemoveItem: (key) => storage.delete(key),
            eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
            focusSessionFromNotification() {{}}, windowTarget: ctx.window, navigatorTarget: ctx.navigator,
            Notification: NotificationCtor, AudioContext: AudioContextCtor, clearTimeout() {{}},
          }},
          eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
          isAppDisposed: () => false,
          api: async (url) => String(url).includes("/api/notifications/feed") ? {{ items: feedItems }} : {{ notifications: {{}}, audio: {{}} }},
          setToast() {{}}, handleAppAuthLoss() {{}}, openSettings() {{}}, closeSettings() {{}},
          resolveAppUrl: (path) => path, versionedShellAssetPath: (path) => path,
          storageGetItem: () => null, storageSetItem() {{}}, storageRemoveItem() {{}}, focusSessionFromNotification() {{}},
          requestFrame: (fn) => fn(), setTimeout: () => 1, clearTimeout() {{}}, setInterval: () => 1, clearInterval() {{}},
        }});
        (async () => {{
          await controller.refreshBackgroundState({{ force: true }});
          const firstDelivered = notifications.splice(0).map((entry) => entry.join(" :: "));
          // Replay the same feed items: none may deliver twice.
          feedItems.push(...{json.dumps(feed_items)});
          await controller.refreshBackgroundState({{ force: true }});
          const replayDelivered = notifications.splice(0).map((entry) => entry.join(" :: "));
          process.stdout.write(JSON.stringify({{
            firstDelivered,
            replayDelivered,
            bellActive: dom.notificationBtn.classList.contains("active"),
            bellTitle: dom.notificationBtn.title,
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_notification_feed_and_read_state_cover_pi_codex_and_claude_code(tmp_path: Path) -> None:
    # Pi, Codex, and Claude Code each publish user-visible completion records.
    # All three must enter the same delivery ledger and feed route with an
    # attention-readable text payload.
    records_by_backend = {
        "pi": [
            {
                "type": "message",
                "id": "pi-final-1",
                "timestamp": "2026-08-04T12:00:00.000Z",
                "message": {
                    "role": "assistant",
                    "stopReason": "end_turn",
                    "content": [{"type": "text", "text": "Pi completed the requested review."}],
                },
            }
        ],
        "codex": [
            {
                "type": "event_msg",
                "ts": 2.0,
                "payload": {
                    "type": "agent_message",
                    "phase": "final_answer",
                    "message": "Codex completed the requested review.",
                },
            }
        ],
        "cc": [
            {
                "type": "assistant",
                "timestamp": "2026-08-04T12:00:03.000Z",
                "message": {
                    "role": "assistant",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "Claude Code completed the requested review."}],
                },
            }
        ],
    }
    expected_texts = {
        "pi": "Pi completed the requested review.",
        "codex": "Codex completed the requested review.",
        "cc": "Claude Code completed the requested review.",
    }

    stop_event = threading.Event()
    stop_event.set()
    with patch("codoxear.voice_push._default_vapid_subject", return_value="mailto:test@example.invalid"):
        coordinator = VoicePushCoordinator(
            app_dir=tmp_path,
            stop_event=stop_event,
            settings_path=tmp_path / "voice_settings.json",
            subscriptions_path=tmp_path / "push_subscriptions.json",
            delivery_ledger_path=tmp_path / "voice_delivery_ledger.json",
            vapid_private_key_path=tmp_path / "vapid.pem",
        )

    for backend, records in records_by_backend.items():
        messages = _extract_delivery_messages(records)
        assert len(messages) == 1
        assert messages[0].message_class == "final_response"
        assert messages[0].text.startswith(expected_texts[backend])
        coordinator.observe_messages(
            session_id=f"{backend}-session",
            session_display_name=backend.title(),
            messages=messages,
        )

    feed_items = _notification_feed_from_route(coordinator)
    assert {item["session_id"] for item in feed_items} == {"pi-session", "codex-session", "cc-session"}
    assert {item["notification_text"] for item in feed_items} == {
        "Pi completed the requested review.",
        "Codex completed the requested review.",
        "Claude Code completed the requested review.",
    }

    delivery = _delivery_after_refresh(feed_items)
    expected_deliveries = {
        "Pi :: Pi completed the requested review.",
        "Codex :: Codex completed the requested review.",
        "Cc :: Claude Code completed the requested review.",
    }
    # Each backend's feed item is delivered exactly once as a desktop
    # notification carrying its session name and notification text; a replay
    # of the same items delivers nothing, and the bell shows the enabled state.
    assert set(delivery["firstDelivered"]) == expected_deliveries
    assert delivery["replayDelivered"] == []
    assert delivery["bellActive"] is True
    assert delivery["bellTitle"] == "Notifications on"
