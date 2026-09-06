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


def _bell_state_after_opening_panel(feed_items: list[dict[str, Any]]) -> dict[str, Any]:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const feedItems = {json.dumps(feed_items)};
        function node() {{
          return {{
            style: {{}}, dataset: {{}}, textContent: "", title: "", disabled: false, open: false, value: "", checked: false,
            _children: [], _attrs: {{}},
            classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
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
          announceBtn: node(), notificationBtn: node(), notificationPanel: node(), notificationList: node(), notificationEmpty: node(),
          notificationClearBtn: node(), notificationEnableBtn: node(), liveAudio: node(),
          voiceSettingsStatus: node(), voiceBaseUrlInput: node(), voiceApiKeyInput: node(), voiceClearApiKeyToggle: node(), narrationSettingToggle: node(),
          voiceSettingsCancelBtn: node(), voiceSettingsSaveBtn: node(),
        }};
        const documentTarget = {{ activeElement: null, contains: () => true, createElement: () => node() }};
        const ctx = {{
          HTMLElement: function HTMLElement() {{}}, console,
          window: {{ isSecureContext: true }}, navigator: {{ userAgent: "X11 Linux" }}, document: documentTarget,
        }};
        vm.createContext(ctx);
        for (const source of [{json.dumps(MODAL_SOURCE)}, {json.dumps(VOICE_HELPERS_SOURCE)}, {json.dumps(NOTIFICATIONS_SOURCE)}, {json.dumps(VOICE_SOURCE)}]) vm.runInContext(source, ctx);
        const controller = ctx.window.CodoxearVoice.createVoiceController({{
          ...dom, windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
          notificationOptions: {{
            root: {{ appendChild() {{}} }}, voiceHost: {{ style: {{}}, appendChild() {{}}, insertBefore() {{}}, firstChild: null }},
            el: (_tag, attrs = {{}}, children = []) => {{ const created = attrs.id && dom[attrs.id] ? dom[attrs.id] : node(); created.append(...children); return created; }}, iconSvg: () => "",
            notificationBtn: dom.notificationBtn, notificationPanel: dom.notificationPanel, notificationList: dom.notificationList,
            notificationEmpty: dom.notificationEmpty, notificationClearBtn: dom.notificationClearBtn, notificationEnableBtn: dom.notificationEnableBtn,
            isAppDisposed: () => false, api: async (url, options = {{}}) => {{
              if (url.includes("/api/notifications/feed")) return {{ items: typeof feeds !== "undefined" ? (feeds.shift() || []) : typeof feedItems !== "undefined" ? feedItems.splice(0) : [] }};
              if (url.includes("/api/notifications/subscription")) return {{ subscriptions: [] }};
              return {{}};
            }}, setToast() {{}}, handleAppAuthLoss() {{}}, resolveAppUrl: (x) => x, versionedShellAssetPath: (x) => x,
            storageGetItem: (key) => key === "codoxear.notificationEnabled" ? "1" : null,
            storageSetItem() {{}}, storageRemoveItem() {{}},
            eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
            focusSessionFromNotification() {{}}, windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
            Notification: undefined, AudioContext: undefined, clearTimeout() {{}},
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
          const beforeOpening = dom.notificationBtn.dataset.unread;
          await dom.notificationBtn.onclick({{ preventDefault() {{}}, stopPropagation() {{}} }});
          process.stdout.write(JSON.stringify({{
            beforeOpening,
            afterOpening: dom.notificationBtn.dataset.unread,
            panelDisplay: dom.notificationPanel.style.display,
            panelRows: dom.notificationList._children.map((row) => row._children[1].textContent),
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_notification_feed_and_read_state_cover_pi_codex_and_claude_code(tmp_path: Path) -> None:
    # Pi intercom messages are custom envelopes, whereas Codex and Claude Code
    # publish user-visible completion records. All three must enter the same
    # delivery ledger and feed route with an attention-readable text payload.
    records_by_backend = {
        "pi": [
            {
                "type": "custom_message",
                "customType": "intercom_message",
                "id": "pi-intercom-1",
                "timestamp": "2026-08-04T12:00:00.000Z",
                "content": "Subagent needs attention in run 12345678-1234-1234-1234-123456789012",
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
        "pi": "Subagent needs attention",
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
        expected_class = "intercom" if backend == "pi" else "final_response"
        assert messages[0].message_class == expected_class
        assert messages[0].text.startswith(expected_texts[backend])
        coordinator.observe_messages(
            session_id=f"{backend}-session",
            session_display_name=backend.title(),
            messages=messages,
        )

    feed_items = _notification_feed_from_route(coordinator)
    assert {item["session_id"] for item in feed_items} == {"pi-session", "codex-session", "cc-session"}
    assert {item["notification_text"] for item in feed_items} == {
        "Subagent needs attention — Subagent (run 12345678)",
        "Codex completed the requested review.",
        "Claude Code completed the requested review.",
    }

    bell_state = _bell_state_after_opening_panel(feed_items)
    assert bell_state["beforeOpening"] == "3"
    assert bell_state["afterOpening"] == "0"
    assert bell_state["panelDisplay"] == "flex"
    assert set(bell_state["panelRows"]) == {
        "Subagent needs attention — Subagent (run 12345678)",
        "Codex completed the requested review.",
        "Claude Code completed the requested review.",
    }
