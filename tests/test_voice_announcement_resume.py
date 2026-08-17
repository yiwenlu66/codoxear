from frontend_module_loader import module_path
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICE_SOURCE = (module_path("app_voice.js")).read_text(encoding="utf-8")
VOICE_HELPERS_SOURCE = (module_path("app_voice_helpers.js")).read_text(encoding="utf-8")
NOTIFICATIONS_SOURCE = (module_path("app_notifications.js")).read_text(encoding="utf-8")
MODAL_SOURCE = (module_path("app_modal.js")).read_text(encoding="utf-8")
POLLING_SOURCE = (module_path("app_polling.js")).read_text(encoding="utf-8")
TRANSCRIPT_SOURCE = (module_path("app_transcript.js")).read_text(encoding="utf-8")
MESSAGE_FLOW_SOURCE = (module_path("app_message_flow.js")).read_text(encoding="utf-8")
SESSION_STATE_SOURCE = (module_path("app_session_state.js")).read_text(encoding="utf-8")
SESSION_CATALOG_SOURCE = module_path("app_session_catalog.js").read_text(encoding="utf-8")


def run_voice_announcement_resume_harness() -> dict:
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const storage = new Map([
          ["codoxear.announcementEnabled", "1"],
          ["codoxear.announcementClientId", "announcement-client"],
        ]);
        const server = {{ listeners: new Set(), heartbeats: [] }};
        const timers = [];
        const streams = [];

        function eventTarget(extra = {{}}) {{
          const listeners = new Map();
          return Object.assign({{
            style: {{}}, classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
            value: "", checked: false, textContent: "", open: false,
            addEventListener(type, listener) {{
              if (!listeners.has(type)) listeners.set(type, new Set());
              listeners.get(type).add(listener);
            }},
            removeEventListener(type, listener) {{
              const entries = listeners.get(type);
              if (entries) entries.delete(listener);
            }},
            emit(type) {{
              for (const listener of listeners.get(type) || []) listener({{ type, preventDefault() {{}} }});
            }},
            setAttribute() {{}}, removeAttribute() {{}}, matches: () => false,
            load() {{}}, pause() {{}}, canPlayType: () => "probably",
          }}, extra);
        }}

        const documentTarget = eventTarget({{
          visibilityState: "hidden",
          activeElement: null,
          contains: () => true,
        }});

        function makeController(ctx) {{
          const liveAudio = eventTarget({{
            src: "", currentSrc: "", currentTime: 0, paused: false, ended: false,
            playCalls: 0,
            play() {{ this.playCalls += 1; return Promise.resolve(); }},
          }});
          const dom = {{
            announceBtn: eventTarget(), notificationBtn: eventTarget(), liveAudio,
            voiceSettingsBackdrop: eventTarget(), voiceSettingsCloseBtn: eventTarget(),
            voiceSettingsStatus: eventTarget(), voiceBaseUrlInput: eventTarget(),
            voiceApiKeyInput: eventTarget(), voiceClearApiKeyToggle: eventTarget(),
            narrationSettingToggle: eventTarget(), voiceSettingsViewer: eventTarget(),
            voiceSettingsCancelBtn: eventTarget(), voiceSettingsSaveBtn: eventTarget(),
          }};
          const controller = ctx.window.CodoxearVoice.createVoiceController({{
            ...dom,
            notificationOptions: {{
              notificationBtn: dom.notificationBtn,
              isAppDisposed: () => false,
              api: async () => ({{ subscriptions: [], items: [] }}),
              setToast() {{}}, handleAppAuthLoss() {{}}, resolveAppUrl: (path) => path, versionedShellAssetPath: (path) => path,
              storageGetItem: (key) => storage && storage.has && storage.has(key) ? storage.get(key) : null,
              storageSetItem() {{}}, storageRemoveItem() {{}},
              eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
              windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
              Notification: ctx.Notification, clearTimeout() {{}},
            }},
            eventBindings: {{ on(target, type, handler) {{ target[`on${{type}}`] = handler; return handler; }} }},
            isAppDisposed: () => false,
            api: (url, options = {{}}) => {{
              if (url === "/api/audio/listener") {{
                const body = options.body || {{}};
                if (body.enabled) server.listeners.add(body.client_id);
                else server.listeners.delete(body.client_id);
                server.heartbeats.push({{ clientId: body.client_id, enabled: Boolean(body.enabled) }});
                return Promise.resolve({{ active_listener_count: server.listeners.size }});
              }}
              if (url === "/api/settings/voice") return Promise.resolve({{
                tts_enabled_for_narration: false,
                tts_enabled_for_final_response: true,
                tts_base_url: "https://api.openai.com/v1",
                has_tts_api_key: true,
                audio: {{ queue_depth: 0, segment_count: 1, last_error: "", stream_url: "/api/audio/live.m3u8" }},
                notifications: {{ enabled_devices: 0, total_devices: 0, vapid_public_key: "" }},
              }});
              return Promise.resolve({{}});
            }},
            setToast() {{}}, handleAppAuthLoss() {{}}, prepareModalOpen() {{}}, afterModalVisibilityChanged() {{}},
            resolveAppUrl: (path) => path, versionedShellAssetPath: (path) => path,
            storageGetItem: (key) => storage.has(key) ? storage.get(key) : null,
            storageSetItem: (key, value) => storage.set(key, String(value)),
            storageRemoveItem: (key) => storage.delete(key),
            requestFrame: (fn) => fn(),
            setTimeout: (callback, delay) => {{ const timer = {{ callback, delay, cleared: false }}; timers.push(timer); return timer; }},
            clearTimeout: (timer) => {{ timer.cleared = true; }},
            setInterval: () => ({{}}), clearInterval() {{}},
            windowTarget: ctx.window, navigatorTarget: ctx.navigator, documentTarget,
          }});
          return {{ controller, liveAudio }};
        }}

        function makeMessageFlow(ctx) {{
          class EventSource {{
            constructor(url) {{ this.url = url; this.listeners = {{}}; streams.push(this); }}
            addEventListener(type, listener) {{ this.listeners[type] = listener; }}
            close() {{ this.closed = true; }}
          }}
          const typingRowRuntime = {{
            snapshot: () => ({{ stats: {{ thinking: 0, thinkingTokens: 0, tools: 0 }} }}),
            updateTypingStats() {{}}, updateSubagentGauge() {{}}, resetTypingStats() {{}},
          }};
          const sessionCatalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({{ consoleError: () => {{}} }});
          sessionCatalog.set("latestSessions", [{{ session_id: "session-a", agent_backend: "pi" }}]);
          const specific = {{
            sessionCatalog,
            sessionState: (() => {{ const store = ctx.window.CodoxearSessionState.createSessionState({{ consoleError: () => {{}} }}); store.set("selected", "session-a"); return store; }})(),
            currentGeneration: () => 1, isAppDisposed: () => false,
            getTurnOpen: () => false, setTurnOpen() {{}},
            activeTranscriptSnapshot: () => ({{ state: "bound", liveCursor: "cursor-a", logPath: "/tmp/session-a.jsonl" }}),
            typingRowRuntime, visibilityState: () => documentTarget.visibilityState,
            navigatorValue: () => ({{ onLine: true }}), EventSource,
            resolveAppUrl: (path) => path, now: () => 0,
            setTimeout: (callback, delay) => {{ const timer = {{ callback, delay, cleared: false }}; timers.push(timer); return timer; }},
            clearTimeout: (timer) => {{ timer.cleared = true; }},
          }};
          const options = new Proxy(specific, {{
            get(target, property) {{ return property in target ? target[property] : (() => {{}}); }},
          }});
          return ctx.window.CodoxearMessageFlow.createMessageFlowController(options);
        }}

        const ctx = {{
          HTMLElement: function HTMLElement() {{}}, console,
          window: {{ isSecureContext: true }},
          navigator: {{ userAgent: "Version/17.0 Safari/605.1.15", vendor: "Apple Computer, Inc." }},
          document: documentTarget,
        }};
        vm.createContext(ctx);
        for (const source of [
          {json.dumps(MODAL_SOURCE)},
          {json.dumps(VOICE_HELPERS_SOURCE)},
          {json.dumps(NOTIFICATIONS_SOURCE)},
          {json.dumps(VOICE_SOURCE)},
          {json.dumps(POLLING_SOURCE)},
          {json.dumps(TRANSCRIPT_SOURCE)},
          {json.dumps(MESSAGE_FLOW_SOURCE)},
          {json.dumps(SESSION_CATALOG_SOURCE)},
          {json.dumps(SESSION_STATE_SOURCE)},
        ]) vm.runInContext(source, ctx);

        (async () => {{
          const first = makeController(ctx);
          const registeredBeforeRestart = server.listeners.size;

          // A server restart drops only the server-side listener set. A new
          // controller must rebuild it from the browser's persisted opt-in.
          server.listeners.clear();
          const resumed = makeController(ctx);
          const registeredByNewController = server.listeners.size;
          await resumed.controller.loadVoiceSettings();

          // The stream was paused while hidden. A visible lifecycle event must
          // restart the live audio path without changing the opt-in.
          resumed.liveAudio.emit("pause");
          const playsBeforeVisible = resumed.liveAudio.playCalls;
          documentTarget.visibilityState = "visible";
          documentTarget.emit("visibilitychange");
          for (const timer of timers.filter((item) => item.delay === 150 && !item.cleared)) await timer.callback();
          const playsAfterVisible = resumed.liveAudio.playCalls;

          // Exercise the real SSE error/retry implementation alongside the
          // same controller. Reconnecting transcript delivery must neither
          // construct a replacement voice state nor clear local opt-in.
          const flow = makeMessageFlow(ctx);
          flow.openMessageEventSource();
          streams[0].onopen();
          streams[0].listeners.error();
          flow.resumeLiveDelivery();

          process.stdout.write(JSON.stringify({{
            registeredBeforeRestart,
            registeredByNewController,
            announcementsAfterRestart: resumed.controller.voiceAnnouncementsEnabled(),
            storageAnnouncementEnabled: storage.get("codoxear.announcementEnabled"),
            playsBeforeVisible,
            playsAfterVisible,
            sseStreams: streams.length,
            announcementsAfterSseReconnect: resumed.controller.voiceAnnouncementsEnabled(),
          }}));
        }})().catch((error) => {{ console.error(error.stack || error); process.exit(1); }});
        """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as script_file:
        script_file.write(script)
        script_file.flush()
        completed = subprocess.run(["node", script_file.name], check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def test_voice_announcements_resume_across_restart_visibility_and_sse_reconnect() -> None:
    result = run_voice_announcement_resume_harness()

    assert result["registeredBeforeRestart"] == 1
    assert result["registeredByNewController"] == 1
    assert result["announcementsAfterRestart"] is True
    assert result["storageAnnouncementEnabled"] == "1"
    assert result["playsAfterVisible"] > result["playsBeforeVisible"]
    assert result["sseStreams"] == 2
    assert result["announcementsAfterSseReconnect"] is True
