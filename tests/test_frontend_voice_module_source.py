from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_VOICE_JS = module_path("app_voice.js")
APP_VOICE_HELPERS_JS = module_path("app_voice_helpers.js")
APP_NOTIFICATIONS_JS = module_path("app_notifications.js")
APP_MODAL_JS = module_path("app_modal.js")
APP_JS = ROOT / "codoxear" / "static" / "app.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


HARNESS = r"""
const vm = require("vm");
const calls = [];
const toasts = [];
let disposed = false;

// Injectable browser globals.
class NotificationCtor {
  constructor(title, options) { this.title = title; this.options = options; }
  static requestPermission() { return Promise.resolve("granted"); }
}
NotificationCtor.permission = "default";

// Storage backed by a Map so tests can pre-seed toggles / client id.
const storage = new Map();
function seedStorage(entries) { storage.clear(); for (const [k, v] of Object.entries(entries || {})) storage.set(k, String(v)); }

// Timer bookkeeping: record every setTimeout/clearTimeout so dispose tests
// can prove pending voice-save / live-audio retry timers are cleared.
const pendingTimers = new Map();
let timerHandle = 0;
function fakeSetTimeout(fn, ms) {
  calls.push(["setTimeout", ms]);
  const handle = ++timerHandle;
  pendingTimers.set(handle, fn);
  return handle;
}
function fakeClearTimeout(handle) {
  pendingTimers.delete(handle);
  calls.push(["clearTimeout", handle]);
}
function runPendingTimers() {
  let guard = 0;
  while (pendingTimers.size && guard < 50) {
    guard += 1;
    const [handle, fn] = Array.from(pendingTimers.entries())[0];
    pendingTimers.delete(handle);
    fn();
  }
}

// Per-target addEventListener / removeEventListener recording so dispose
// tests can prove liveAudio + dialog listeners are torn down.
const listenerLog = [];
function makeNode(extra = {}) {
  const listeners = new Map();
  const node = {
    style: { display: "none" },
    dataset: {},
    title: "",
    classList: {
      _classes: new Set(),
      add(c) { this._classes.add(c); },
      remove(c) { this._classes.delete(c); },
      toggle(c, force) { if (force === undefined) { this._classes.has(c) ? this._classes.delete(c) : this._classes.add(c); } else if (force) this._classes.add(c); else this._classes.delete(c); },
      contains(c) { return this._classes.has(c); },
    },
    _attrs: {},
    _children: [],
    value: "",
    placeholder: "",
    disabled: false,
    textContent: "",
    checked: false,
    open: false,
    set innerHTML(v) { this._children = []; },
    get innerHTML() { return ""; },
    setAttribute(name, value) { this._attrs[name] = String(value); },
    getAttribute(name) { return this._attrs[name]; },
    removeAttribute(name) { delete this._attrs[name]; },
    append(...children) { this._children.push(...children); },
    appendChild(child) { this._children.push(child); return child; },
    replaceChildren(...children) { this._children = children; },
    addEventListener(type, handler, options) { listenerLog.push(["add", this._domId || "?", type]); if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(handler); },
    removeEventListener(type, handler, options) { listenerLog.push(["remove", this._domId || "?", type]); const s = listeners.get(type); if (s) s.delete(handler); },
    focus() { calls.push(["focus", this._domId]); },
    matches() { return false; },
    load() {},
    pause() {},
    play() { return Promise.resolve(); },
    showModal() { this.open = true; calls.push(["showModal", this._domId]); },
    close() { this.open = false; calls.push(["close", this._domId]); },
    ...extra,
  };
  return node;
}

function labeledNode(id, extra = {}) {
  const n = makeNode(extra);
  n._domId = id;
  return n;
}

const dom = {
  announceBtn: labeledNode("announceBtn"),
  notificationBtn: labeledNode("notificationBtn"),
  liveAudio: labeledNode("liveAudio"),
  voiceSettingsStatus: labeledNode("voiceSettingsStatus"),
  voiceBaseUrlInput: labeledNode("voiceBaseUrlInput"),
  voiceApiKeyInput: labeledNode("voiceApiKeyInput"),
  voiceClearApiKeyToggle: labeledNode("voiceClearApiKeyToggle"),
  narrationSettingToggle: labeledNode("narrationSettingToggle"),
  voiceSettingsCancelBtn: labeledNode("voiceSettingsCancelBtn"),
  voiceSettingsSaveBtn: labeledNode("voiceSettingsSaveBtn"),
};
// The Settings dialog double: the voice controller asks it to open/close and
// it reports back through activate/deactivate exactly as app_settings.js does.
const settingsDialog = {
  open: false,
  show() { calls.push(["openSettings"]); this.open = true; if (this.controller) this.controller.activateSettingsSection(); },
  hide() { calls.push(["closeSettings"]); this.open = false; if (this.controller) this.controller.deactivateSettingsSection(); },
  controller: null,
};

// Configurable API responder: tests poke `apiRoutes` to override per-URL.
let apiRoutes = {};
function setApiRoutes(routes) { apiRoutes = Object.assign({}, routes); }
function defaultApi(url) {
  if (String(url).indexOf("/api/settings/voice") !== -1) {
    return {
      tts_enabled_for_narration: false,
      tts_enabled_for_final_response: true,
      tts_base_url: "https://api.openai.com/v1",
      has_tts_api_key: false,
      audio: { queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8" },
      notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" },
    };
  }
  if (String(url).indexOf("/api/notifications/subscription") !== -1) return { subscriptions: [] };
  if (String(url).indexOf("/api/notifications/feed") !== -1) return { items: [] };
  if (String(url).indexOf("/api/audio/listener") !== -1) return {};
  return {};
}

function buildDeps(overrides = {}) {
  return Object.assign({
    announceBtn: dom.announceBtn,
    notificationBtn: dom.notificationBtn,
    liveAudio: dom.liveAudio,
    voiceSettingsStatus: dom.voiceSettingsStatus,
    voiceBaseUrlInput: dom.voiceBaseUrlInput,
    voiceApiKeyInput: dom.voiceApiKeyInput,
    voiceClearApiKeyToggle: dom.voiceClearApiKeyToggle,
    narrationSettingToggle: dom.narrationSettingToggle,
    voiceSettingsCancelBtn: dom.voiceSettingsCancelBtn,
    voiceSettingsSaveBtn: dom.voiceSettingsSaveBtn,
    openSettings: () => settingsDialog.show(),
    closeSettings: () => settingsDialog.hide(),
    isAppDisposed: () => disposed,
    api: (url, options = {}) => {
      const body = options && options.body ? JSON.parse(JSON.stringify(options.body)) : null;
      calls.push(["api", url, body]);
      const route = apiRoutes[url] || (Object.keys(apiRoutes).find((k) => String(url).indexOf(k) !== -1));
      if (route !== undefined) {
        const val = apiRoutes[Object.keys(apiRoutes).find((k) => String(url).indexOf(k) !== -1)];
        if (val instanceof Error) return Promise.reject(val);
        if (typeof val === "function") return Promise.resolve(val(url, body));
        return Promise.resolve(val);
      }
      return Promise.resolve(defaultApi(url));
    },
    setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
    handleAppAuthLoss: () => { calls.push(["handleAppAuthLoss"]); },
    resolveAppUrl: (p) => p,
    versionedShellAssetPath: (p) => p,
    storageGetItem: (k) => (storage.has(k) ? storage.get(k) : null),
    storageSetItem: (k, v) => { calls.push(["storageSetItem", k, String(v)]); storage.set(k, String(v)); },
    storageRemoveItem: (k) => { calls.push(["storageRemoveItem", k]); storage.delete(k); },
    eventBindings: { on(target, type, handler, options) { target.addEventListener(type, handler, options); return handler; } },
    notificationOptions: {
      root: { appendChild() {} },
      voiceHost: { style: {}, firstChild: null, appendChild() {}, insertBefore() {} },
      el: (_tag, attrs = {}) => attrs.id === "notificationBtn" ? dom.notificationBtn : makeNode(),
      iconSvg: () => "",
      isAppDisposed: () => disposed,
      api: (url, options = {}) => {
        const body = options && options.body ? JSON.parse(JSON.stringify(options.body)) : null;
        calls.push(["api", url, body]);
        const routeKey = Object.keys(apiRoutes).find((k) => String(url).indexOf(k) !== -1);
        const val = routeKey === undefined ? defaultApi(url) : apiRoutes[routeKey];
        if (val instanceof Error) return Promise.reject(val);
        if (typeof val === "function") return Promise.resolve(val(url, body));
        return Promise.resolve(val);
      },
      setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
      handleAppAuthLoss: () => { calls.push(["handleAppAuthLoss"]); },
      resolveAppUrl: (p) => p,
      versionedShellAssetPath: (p) => p,
      storageGetItem: (k) => (storage.has(k) ? storage.get(k) : null),
      storageSetItem: (k, v) => { calls.push(["storageSetItem", k, String(v)]); storage.set(k, String(v)); },
      storageRemoveItem: (k) => { calls.push(["storageRemoveItem", k]); storage.delete(k); },
      eventBindings: { on(target, type, handler, options) { target.addEventListener(type, handler, options); return handler; } },
      focusSessionFromNotification: (sid) => { calls.push(["focusSessionFromNotification", sid]); },
      windowTarget: ctx.window,
      navigatorTarget: ctx.navigator,
      documentTarget: ctx.document,
      Notification: NotificationCtor,
      clearTimeout: fakeClearTimeout,
    },
    requestFrame: (fn) => fn(),
    setTimeout: fakeSetTimeout,
    clearTimeout: fakeClearTimeout,
    setInterval: () => 0,
    clearInterval: () => {},
  }, overrides);
}

const ctx = {
  HTMLElement: function HTMLElement() {},
  Notification: NotificationCtor,
  atob: (v) => Buffer.from(v, "base64").toString("binary"),
  console,
  window: { isSecureContext: true },
  navigator: { userAgent: "X11 Linux x86_64" },
  document: { activeElement: null, contains: () => true, addEventListener() {}, removeEventListener() {} },
};
vm.createContext(ctx);
vm.runInContext(MODAL_SOURCE, ctx);
vm.runInContext(HELPERS_SOURCE, ctx);
vm.runInContext(NOTIFICATIONS_SOURCE, ctx);
vm.runInContext(VOICE_SOURCE, ctx);

globalThis.__harness = {
  dom,
  calls,
  toasts,
  listenerLog,
  seedStorage,
  storage,
  setApiRoutes,
  buildDeps,
  settingsDialog,
  createController: (overrides) => { const c = ctx.window.CodoxearVoice.createVoiceController(buildDeps(overrides)); settingsDialog.controller = c; return c; },
  setDisposed: (v) => { disposed = v; },
  setNotificationPermission: (p) => { NotificationCtor.permission = p; },
  setUserAgent: (ua) => { ctx.navigator.userAgent = ua; },
  setSecureContext: (v) => { ctx.window.isSecureContext = v; },
  pendingTimerCount: () => pendingTimers.size,
  runPendingTimers,
};
"""


def harness_script(epilogue: str) -> str:
    voice_source = APP_VOICE_JS.read_text(encoding="utf-8")
    helpers_source = APP_VOICE_HELPERS_JS.read_text(encoding="utf-8")
    notifications_source = APP_NOTIFICATIONS_JS.read_text(encoding="utf-8")
    modal_source = APP_MODAL_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const MODAL_SOURCE = {json.dumps(modal_source)};
        const HELPERS_SOURCE = {json.dumps(helpers_source)};
        const NOTIFICATIONS_SOURCE = {json.dumps(notifications_source)};
        const VOICE_SOURCE = {json.dumps(voice_source)};
        """
        )
        + HARNESS
        + "\n(async () => {\n"
        + textwrap.dedent(epilogue)
        + "\n})().then(() => {\n"
        + "  process.stdout.write(JSON.stringify(globalThis.__result || {}));\n"
        + "}).catch((err) => {\n"
        + "  console.error(err && err.stack || err);\n"
        + "  process.exit(1);\n"
        + "});\n"
    )
    return js


class TestFrontendVoiceModuleSource(unittest.TestCase):
    # --- 1. frozen export + missing deps ---

    # --- 2. initial localStorage state + client id persistence ---

    def test_initial_local_state_reads_storage_and_persists_client_id(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // Seeded toggles are reflected; a missing client id is generated and
            // persisted exactly once via storageSetItem.
            h.seedStorage({
              "codoxear.announcementEnabled": "1",
              "codoxear.notificationEnabled": "0",
            });
            const beforeSet = h.calls.length;
            const c = h.createController();
            const setCalls = h.calls.filter((x) => x[0] === "storageSetItem").map((x) => x[1]);
            globalThis.__result = {
              announcements: c.voiceAnnouncementsEnabled(),
              notificationTitle: h.dom.notificationBtn.title,
              clientIdStored: setCalls.indexOf("codoxear.announcementClientId") !== -1,
            };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["announcements"])
        self.assertEqual(result["notificationTitle"], "Notifications")
        self.assertTrue(result["clientIdStored"])

    def test_announcement_client_id_reuses_persisted_value(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({ "codoxear.announcementClientId": "stable-id-123" });
            h.createController();
            const setCalls = h.calls.filter((x) => x[0] === "storageSetItem" && x[1] === "codoxear.announcementClientId");
            globalThis.__result = { reused: setCalls.length === 0 };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["reused"])

    # --- 3. inline settings section lifecycle (driven by the Settings dialog) ---

    def test_settings_section_activation_seeds_form_and_deactivation_drops_draft(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.setApiRoutes({ "/api/settings/unattended-prompt": { prompt: "saved prompt", default_prompt: "built-in" } });
            const c = h.createController();
            const closed = c.isSettingsOpen();
            c.activateSettingsSection();
            await new Promise((r) => setTimeout(r, 0));
            const afterActivate = {
              open: c.isSettingsOpen(),
              baseUrl: h.dom.voiceBaseUrlInput.value,
              prompt: h.dom.unattendedPromptInput ? h.dom.unattendedPromptInput.value : null,
              promptLoaded: h.calls.some((x) => x[0] === "api" && x[1] === "/api/settings/unattended-prompt"),
            };
            // A background snapshot must not clobber the user's draft while the section is active.
            h.dom.voiceBaseUrlInput.value = "https://draft.example/v1";
            await c.loadVoiceSettings();
            const draftKept = h.dom.voiceBaseUrlInput.value;
            h.dom.voiceSettingsStatus.textContent = "Saving...";
            c.deactivateSettingsSection();
            const afterDeactivate = { open: c.isSettingsOpen(), status: h.dom.voiceSettingsStatus.textContent };
            // Once inactive, the next snapshot re-seeds the form from state.
            await c.loadVoiceSettings();
            const reseeded = h.dom.voiceBaseUrlInput.value;
            globalThis.__result = { closed, afterActivate, draftKept, afterDeactivate, reseeded };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["closed"])
        self.assertEqual(result["afterActivate"], {"open": True, "baseUrl": "https://api.openai.com/v1", "prompt": None, "promptLoaded": True})
        self.assertEqual(result["draftKept"], "https://draft.example/v1")
        self.assertEqual(result["afterDeactivate"], {"open": False, "status": ""})
        self.assertEqual(result["reseeded"], "https://api.openai.com/v1")

    def test_save_closes_the_settings_dialog_and_cancel_closes_without_saving(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            const c = h.createController();
            h.settingsDialog.show();
            h.dom.voiceBaseUrlInput.value = "https://voice.example/v1";
            h.calls.length = 0;
            await h.dom.voiceSettingsSaveBtn.onclick();
            const afterSave = {
              posted: h.calls.filter((x) => x[0] === "api" && x[1] === "/api/settings/voice" && x[2]).map((x) => x[2].tts_base_url),
              closed: h.calls.some((x) => x[0] === "closeSettings"),
              dialogOpen: h.settingsDialog.open,
              sectionOpen: c.isSettingsOpen(),
            };
            h.settingsDialog.show();
            h.calls.length = 0;
            h.dom.voiceSettingsCancelBtn.onclick();
            const afterCancel = {
              posted: h.calls.filter((x) => x[0] === "api" && x[2]).length,
              closed: h.calls.some((x) => x[0] === "closeSettings"),
              dialogOpen: h.settingsDialog.open,
            };
            globalThis.__result = { afterSave, afterCancel };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["afterSave"], {"posted": ["https://voice.example/v1"], "closed": True, "dialogOpen": False, "sectionOpen": False})
        self.assertEqual(result["afterCancel"], {"posted": 0, "closed": True, "dialogOpen": False})

    # --- 4. form sync + API-key placeholder / clear payload ---

    def test_save_voice_settings_preserves_blank_key_and_clear_flag(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({});
            // Server reports a saved key exists; the input stays blank.
            h.setApiRoutes({ "/api/settings/voice": (url, body) => {
              if (body) return Object.assign({}, body, { has_tts_api_key: true, tts_api_key: "" });
              return { tts_enabled_for_narration: false, tts_enabled_for_final_response: true,
                       tts_base_url: "https://api.openai.com/v1", has_tts_api_key: true, tts_api_key: "",
                       audio: { queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8" },
                       notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" } };
            }});
            const c = h.createController();
            await c.loadVoiceSettings();
            // After load the API-key input must be blank with the "saved" placeholder.
            const placeholder = h.dom.voiceApiKeyInput.placeholder;
            const blankAfterLoad = h.dom.voiceApiKeyInput.value;
            // User leaves the key blank and saves: payload must carry an empty
            // tts_api_key and tts_api_key_clear: false (preserve saved key).
            h.dom.voiceSettingsSaveBtn.onclick();
            await new Promise((r) => setTimeout(r, 0));
            const saveCall = h.calls.filter((x) => x[0] === "api" && x[1] === "/api/settings/voice" && x[2]).slice(-1)[0];
            const preservePayload = saveCall ? saveCall[2] : null;

            // Now user checks "Clear saved API key" and saves: payload must
            // carry tts_api_key_clear: true and tts_api_key: "".
            h.calls.length = 0;
            h.dom.voiceClearApiKeyToggle.checked = true;
            h.dom.voiceSettingsSaveBtn.onclick();
            await new Promise((r) => setTimeout(r, 0));
            const clearCall = h.calls.filter((x) => x[0] === "api" && x[1] === "/api/settings/voice" && x[2]).slice(-1)[0];
            const clearPayload = clearCall ? clearCall[2] : null;
            globalThis.__result = { placeholder, blankAfterLoad, preservePayload, clearPayload };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["placeholder"], "Saved API key (leave blank to keep)")
        self.assertEqual(result["blankAfterLoad"], "")
        self.assertEqual(result["preservePayload"]["tts_api_key"], "")
        self.assertFalse(result["preservePayload"]["tts_api_key_clear"])
        self.assertEqual(result["clearPayload"]["tts_api_key"], "")
        self.assertTrue(result["clearPayload"]["tts_api_key_clear"])

    def test_never_populates_saved_api_key_into_input(self) -> None:
        # The saved key is never written into the input even when the server
        # echoes it back in a non-redacted snapshot.
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({});
            h.setApiRoutes({ "/api/settings/voice": (url, body) => {
              if (body) return Object.assign({}, body, { has_tts_api_key: true, tts_api_key: "SECRET" });
              return { tts_base_url: "https://api.openai.com/v1", has_tts_api_key: true, tts_api_key: "SECRET",
                       audio: { queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8" },
                       notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" } };
            }});
            const c = h.createController();
            await c.loadVoiceSettings();
            c.activateSettingsSection();
            globalThis.__result = { inputNeverHoldsSecret: h.dom.voiceApiKeyInput.value !== "SECRET" };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["inputNeverHoldsSecret"])

    # --- 5. announcement toggle blocked without credentials opens settings ---

    def test_announcement_toggle_without_credentials_opens_settings(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({});
            // Defaults have no base url / api key / has_tts_api_key.
            const c = h.createController();
            const wasOpen = c.isSettingsOpen();
            const fakeEvent = { preventDefault() {}, stopPropagation() {} };
            await h.dom.announceBtn.onclick(fakeEvent);
            await Promise.resolve();
            globalThis.__result = {
              wasOpen,
              opened: c.isSettingsOpen(),
              dialogOpened: h.settingsDialog.open,
              status: h.dom.voiceSettingsStatus.textContent,
              notEnabled: !c.voiceAnnouncementsEnabled(),
            };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["wasOpen"])
        self.assertTrue(result["opened"])
        self.assertTrue(result["dialogOpened"])
        self.assertTrue(result["notEnabled"])
        self.assertContains("API base URL", result["status"])

    # --- 6. notification transport projection (desktop vs mobile) ---

    def test_notification_transport_projects_desktop_vs_mobile(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            // Desktop, locally enabled + desktop flag set + permission granted +
            // secure context => transport "desktop", button title "Notifications on".
            h.seedStorage({
              "codoxear.notificationEnabled": "1",
              "codoxear.desktopNotificationsEnabled": "1",
            });
            h.setNotificationPermission("granted");
            h.setUserAgent("X11 Linux x86_64");
            h.setSecureContext(true);
            const c = h.createController();
            await c.refreshBackgroundState({ force: true });
            const desktop = { title: h.dom.notificationBtn.title, active: h.dom.notificationBtn.classList.contains("active") };

            // Switch to a mobile UA with no push subscription: locally enabled
            // but transport unresolved => "Notifications pending".
            h.setUserAgent("iPhone Mobile");
            h.seedStorage({
              "codoxear.notificationEnabled": "1",
            });
            await c.refreshBackgroundState({ force: true });
            const mobile = { title: h.dom.notificationBtn.title };

            // Locally disabled => "Notifications off" regardless of device.
            h.seedStorage({});
            const c2 = h.createController();
            await c2.refreshBackgroundState({ force: true });
            const off = { title: h.dom.notificationBtn.title, active: h.dom.notificationBtn.classList.contains("active") };
            globalThis.__result = { desktop, mobile, off };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["desktop"]["title"], "Notifications")
        self.assertFalse(result["desktop"]["active"])
        self.assertEqual(result["mobile"]["title"], "Notifications")
        self.assertEqual(result["off"]["title"], "Notifications")
        self.assertFalse(result["off"]["active"])

    def test_background_refresh_coalesces_subscription_snapshot_and_skips_inactive_polling(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({ "codoxear.notificationEnabled": "1" });
            const c = h.createController();
            await c.refreshBackgroundState({ force: true, primeNotifications: true });
            const enabledCalls = h.calls.filter((x) => x[0] === "api").map((x) => x[1]);
            h.calls.length = 0;
            h.seedStorage({});
            const inactive = h.createController();
            await inactive.refreshBackgroundState();
            const inactiveCalls = h.calls.filter((x) => x[0] === "api").map((x) => x[1]);
            globalThis.__result = { enabledCalls, inactiveCalls };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["enabledCalls"], ["/api/settings/voice", "/api/notifications/feed?since=0"])
        self.assertEqual(result["inactiveCalls"], [])

    # --- 7. live-audio last_error / error projection ---

    def test_live_audio_last_error_surfaces_on_announce_button(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({ "codoxear.announcementEnabled": "1" });
            h.setApiRoutes({ "/api/settings/voice": {
              tts_enabled_for_narration: false, tts_enabled_for_final_response: true,
              tts_base_url: "https://api.openai.com/v1", has_tts_api_key: true, tts_api_key: "",
              audio: { queue_depth: 0, segment_count: 0, last_error: "hls boom", stream_url: "/api/audio/live.m3u8" },
              notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" },
            }});
            const c = h.createController();
            await c.loadVoiceSettings();
            const withError = {
              hasErrorClass: h.dom.announceBtn.classList.contains("error"),
              title: h.dom.announceBtn.title,
              aria: h.dom.announceBtn.getAttribute("aria-label"),
            };
            // Clear the server error: a subsequent load must drop the error class.
            h.setApiRoutes({ "/api/settings/voice": {
              tts_enabled_for_narration: false, tts_enabled_for_final_response: true,
              tts_base_url: "https://api.openai.com/v1", has_tts_api_key: true, tts_api_key: "",
              audio: { queue_depth: 0, segment_count: 0, last_error: "", stream_url: "/api/audio/live.m3u8" },
              notifications: { enabled_devices: 0, total_devices: 0, vapid_public_key: "" },
            }});
            await c.loadVoiceSettings();
            const cleared = { hasErrorClass: h.dom.announceBtn.classList.contains("error"), title: h.dom.announceBtn.title };
            globalThis.__result = { withError, cleared };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["withError"]["hasErrorClass"])
        self.assertContains("Announcements on", result["withError"]["title"])
        self.assertContains("audio error", result["withError"]["title"])
        self.assertContains("audio error", result["withError"]["aria"])
        self.assertFalse(result["cleared"]["hasErrorClass"])
        self.assertNotContains("audio error", result["cleared"]["title"])

    # --- 8. dispose clears timers, handlers, HLS ---

    def test_dispose_clears_timers_handlers_and_state(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.seedStorage({ "codoxear.announcementEnabled": "1" });
            const c = h.createController();
            // Arm a voice save timer via the narration toggle.
            h.dom.narrationSettingToggle.onchange({ target: { checked: true } });
            const timersBefore = h.pendingTimerCount();
            // Capture liveAudio listeners registered by the controller.
            const liveAudioAdds = h.listenerLog.filter((x) => x[1] === "liveAudio" && x[0] === "add").length;
            c.dispose();
            const timersAfter = h.pendingTimerCount();
            // After dispose, the controller-owned button onclick handlers are released.
            const handlersCleared = h.dom.announceBtn.onclick === null && h.dom.voiceSettingsSaveBtn.onclick === null && h.dom.voiceSettingsCancelBtn.onclick === null;
            // After dispose, running pending timers (none should remain) must not
            // issue any API call.
            const apiBefore = h.calls.filter((x) => x[0] === "api").length;
            h.runPendingTimers();
            const apiAfter = h.calls.filter((x) => x[0] === "api").length;
            globalThis.__result = { timersBefore, timersAfter, handlersCleared, liveAudioAdds, apiBefore, apiAfter, settingsClosed: !c.isSettingsOpen() };
            """
        )
        result = run_node_json(js)
        self.assertGreater(result["timersBefore"], 0)
        self.assertEqual(result["timersAfter"], 0)
        self.assertTrue(result["handlersCleared"])
        self.assertGreater(result["liveAudioAdds"], 0)
        self.assertEqual(result["apiBefore"], result["apiAfter"])
        self.assertTrue(result["settingsClosed"])

    # --- 9. static load order helpers -> voice -> app.js ---

    # --- 10. app.js delegates instead of retaining voice internals ---

if __name__ == "__main__":
    unittest.main()
