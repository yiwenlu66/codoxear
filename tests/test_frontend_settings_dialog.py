"""Settings dialog: modal lifecycle, live appearance controls, inline voice section.

Executes app_settings.js against the real app_theme.js controller in Node's
vm with a small DOM double, and asserts the dialog renders controller state
(swatch/mode selection, custom CSS), forwards intent to the controller (family,
mode, debounced custom CSS), never closes on Escape, mounts the voice
section it is handed below Appearance, and reports its visibility to that
section's owner (activate after show, deactivate before hide).
"""

from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap


APP_MODAL_JS = module_path("app_modal.js")
APP_THEME_JS = module_path("app_theme.js")
APP_SETTINGS_JS = module_path("app_settings.js")

HARNESS = """
class ElementStub {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.attrs = {};
    this.children = [];
    this.parentNode = null;
    this.textContent = "";
    this.innerHTML = "";
    this.value = "";
    this.style = {};
    this.listeners = {};
    this.open = false;
    this.disabled = false;
    this.isConnected = true;
    const classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      toggle: (name, force) => { if (force === undefined ? !classes.has(name) : force) classes.add(name); else classes.delete(name); return classes.has(name); },
      contains: (name) => classes.has(name),
    };
    Object.defineProperty(this, "className", { get: () => [...classes].join(" "), set: (text) => { classes.clear(); String(text).split(/\\s+/).filter(Boolean).forEach((name) => classes.add(name)); } });
  }
  get id() { return this.attrs.id || ""; }
  set id(value) { if (value) this.attrs.id = String(value); else delete this.attrs.id; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; }
  getAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attrs, name) ? this.attrs[name] : null; }
  appendChild(child) { if (child.parentNode) child.parentNode.removeChild(child); child.parentNode = this; this.children.push(child); return child; }
  removeChild(child) { const i = this.children.indexOf(child); if (i >= 0) this.children.splice(i, 1); child.parentNode = null; return child; }
  insertBefore(child, ref) { const i = ref ? this.children.indexOf(ref) : -1; if (i < 0) return this.appendChild(child); child.parentNode = this; this.children.splice(i, 0, child); return child; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  emit(type, event = {}) { const e = { type, target: this, currentTarget: this, preventDefault() { e.defaultPrevented = true; }, stopPropagation() {}, ...event }; for (const h of this.listeners[type] || []) h(e); return e; }
  showModal() { this.open = true; calls.push("showModal"); }
  close() { this.open = false; calls.push("close"); }
  focus() { calls.push(`focus:${this.attrs.id || this.attrs["data-theme-family"] || this.textContent}`); }
  find(predicate) { for (const child of this.children) { if (predicate(child)) return child; const hit = child.find(predicate); if (hit) return hit; } return null; }
  findAll(predicate, out = []) { for (const child of this.children) { if (predicate(child)) out.push(child); child.findAll(predicate, out); } return out; }
}
const calls = [];
function el(tag, attrs = {}, children = []) {
  const node = new ElementStub(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v; else if (k === "html") node.innerHTML = v; else node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}
const html = new ElementStub("html");
const head = new ElementStub("head");
html.appendChild(head);
const documentTarget = {
  documentElement: html, head, activeElement: null,
  createElement: (tag) => new ElementStub(tag),
  querySelector: () => null,
  getElementById: (id) => html.find((n) => n.attrs.id === id),
  contains: () => true,
};
const storage = {};
const darkQuery = { matches: false, listeners: [], addEventListener(t, h) { this.listeners.push(h); }, removeEventListener() {} };
const themeController = CodoxearTheme.createThemeController({
  documentTarget,
  storageGetItem: (k) => (k in storage ? storage[k] : null),
  storageSetItem: (k, v) => { storage[k] = String(v); },
  storageRemoveItem: (k) => { delete storage[k]; },
  matchMedia: () => darkQuery,
  versionedAssetPath: (p) => p,
});
const timers = [];
const root = new ElementStub("div");
const opener = el("button", { id: "settingsBtnSide" });
const voiceCalls = [];
// The voice section is built by app_voice.js; here a stand-in carries the
// same id and one text entry so mount order and lifecycle can be observed.
const voiceSection = el("section", { class: "settingsSection", id: "voiceSettingsSection" }, [el("input", { id: "voiceBaseUrlInput", type: "text" })]);
const controller = CodoxearSettings.createSettingsDialogController({
  root, el, iconSvg: (name) => `<svg data-icon="${name}"></svg>`,
  themeController, openButton: opener,
  voiceSection,
  activateVoiceSection: () => { voiceCalls.push("activate"); calls.push("voice:activate"); },
  deactivateVoiceSection: () => { voiceCalls.push("deactivate"); calls.push("voice:deactivate"); },
  documentTarget, ElementCtor: ElementStub,
  prepareModalOpen: () => calls.push("prepare"),
  afterModalVisibilityChanged: () => calls.push("visibility"),
  addEvent: (target, type, handler) => target.addEventListener(type, handler),
  setTimeout: (fn, ms) => { const id = timers.length + 1; timers.push({ id, fn, ms }); return id; },
  clearTimeout: (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); },
  focusModalCloseButton: (_viewer, button) => button.focus(),
  isModalTargetOpen: (target) => target.open || (target.style.display && target.style.display !== "none"),
  restoreModalFocus: (target, stillOpen) => { calls.push(`restore:${stillOpen()}`); if (target) target.focus(); },
});
const viewer = controller.viewer;
const family = (name) => viewer.find((n) => n.attrs["data-theme-family"] === name);
const mode = (name) => viewer.find((n) => n.attrs["data-theme-mode"] === name);
const swatches = viewer.find((n) => n.classList.contains("themeSwatches"));
const textarea = viewer.find((n) => n.attrs.id === "settingsCustomCss");
const hint = viewer.find((n) => n.attrs.id === "settingsModeHint");
const selection = () => ({
  family: ["paper", "clay", "slate"].filter((f) => family(f).classList.contains("active")),
  mode: ["system", "light", "dark"].filter((m) => mode(m).classList.contains("active")),
  swatchMode: swatches.attrs["data-swatch-mode"],
  checked: ["paper", "clay", "slate"].map((f) => family(f).attrs["aria-checked"]).join(""),
});
function runTimers() { while (timers.length) { const t = timers.shift(); t.fn(); } }
"""


def run(body: str) -> dict:
    sources = [p.read_text(encoding="utf-8") for p in (APP_MODAL_JS, APP_THEME_JS, APP_SETTINGS_JS)]
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const ctx = { window: {}, requestAnimationFrame: (cb) => cb() };
        vm.createContext(ctx);
        for (const source of __SOURCES__) vm.runInContext(source, ctx);
        const CodoxearTheme = ctx.window.CodoxearTheme;
        const CodoxearSettings = ctx.window.CodoxearSettings;
        __HARNESS__
        const out = (function () { __BODY__ })();
        process.stdout.write(JSON.stringify(out));
        """
    ).replace("__SOURCES__", json.dumps(sources)).replace("__HARNESS__", HARNESS).replace("__BODY__", body)
    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(result.stdout)


def test_dialog_is_a_form_viewer_with_appearance_then_voice_sections() -> None:
    data = run("""
    const buttons = viewer.findAll((n) => n.tagName === "BUTTON").map((n) => n.attrs.id || n.attrs["data-theme-family"] || n.attrs["data-theme-mode"]);
    const body = viewer.find((n) => n.classList.contains("formBody"));
    return {
      className: viewer.className, label: viewer.attrs["aria-label"], title: viewer.find((n) => n.classList.contains("title")).textContent,
      buttons, mounted: root.children.map((n) => n.attrs.id), initial: selection(), hint: hint.textContent,
      swatchPreviews: viewer.findAll((n) => n.attrs["data-swatch-family"]).length,
      sections: body.children.map((n) => [n.tagName, n.attrs.id, n.find((c) => c.classList.contains("settingsSectionTitle"))?.textContent || null]),
      voiceMountedInBody: voiceSection.parentNode === body,
      frozen: Object.isFrozen(controller),
    };
    """)
    assert data["className"] == "formViewer formDialog"
    assert data["label"] == "Settings"
    assert data["title"] == "Settings"
    # Appearance is reversible through the swatches, mode chips, and clearing
    # the CSS field; there is no dedicated reset control.
    assert data["buttons"] == ["settingsCloseBtn", "paper", "clay", "slate", "system", "light", "dark"]
    assert data["sections"] == [["SECTION", "appearanceSettingsSection", "Appearance"], ["SECTION", "voiceSettingsSection", None]]
    assert data["voiceMountedInBody"] is True
    assert data["mounted"] == ["settingsBackdrop", "settingsViewer"]
    assert data["initial"] == {"family": ["paper"], "mode": ["system"], "swatchMode": "light", "checked": "truefalsefalse"}
    assert data["hint"] == "Follows the system setting (currently light)."
    assert data["swatchPreviews"] == 3
    assert data["frozen"] is True


def test_open_button_shows_modal_and_close_restores_focus() -> None:
    data = run("""
    opener.emit("click");
    const openSnapshot = { open: controller.isOpen(), backdrop: root.children[0].style.display, viewer: viewer.style.display, native: viewer.open };
    viewer.find((n) => n.attrs.id === "settingsCloseBtn").emit("click");
    return { openSnapshot, closed: controller.isOpen(), calls };
    """)
    assert data["openSnapshot"] == {"open": True, "backdrop": "block", "viewer": "flex", "native": True}
    assert data["closed"] is False
    # The voice section is activated only once the modal is visible and
    # focused, and deactivated before the dialog starts hiding.
    assert data["calls"] == ["prepare", "showModal", "visibility", "focus:settingsCloseBtn", "voice:activate", "voice:deactivate", "close", "visibility", "restore:false", "focus:settingsBtnSide"]


def test_escape_cancel_is_swallowed_and_backdrop_click_closes() -> None:
    data = run("""
    controller.show();
    const cancel = viewer.emit("cancel");
    const afterEscape = controller.isOpen();
    viewer.emit("click", { target: viewer });
    return { prevented: cancel.defaultPrevented === true, afterEscape, afterBackdrop: controller.isOpen() };
    """)
    assert data == {"prevented": True, "afterEscape": True, "afterBackdrop": False}


def test_swatch_and_mode_buttons_apply_live_and_render_selection() -> None:
    data = run("""
    controller.show();
    family("slate").emit("click");
    const afterFamily = { ...selection(), attrs: { ...html.attrs }, hint: hint.textContent };
    mode("dark").emit("click");
    const afterMode = { ...selection(), attrs: { ...html.attrs }, hint: hint.textContent, storage: { ...storage } };
    darkQuery.matches = true;
    mode("system").emit("click");
    for (const h of darkQuery.listeners) h({ matches: true });
    const afterSystem = { ...selection(), attrs: { ...html.attrs }, hint: hint.textContent };
    return { afterFamily, afterMode, afterSystem };
    """)
    assert data["afterFamily"] == {"family": ["slate"], "mode": ["system"], "swatchMode": "light", "checked": "falsefalsetrue", "attrs": {"data-theme": "slate", "data-mode": "light"}, "hint": "Follows the system setting (currently light)."}
    assert data["afterMode"] == {"family": ["slate"], "mode": ["dark"], "swatchMode": "dark", "checked": "falsefalsetrue", "attrs": {"data-theme": "slate", "data-mode": "dark"}, "hint": "Always dark.", "storage": {"codoxear.ui.theme.family": "slate", "codoxear.ui.theme.mode": "dark"}}
    assert data["afterSystem"] == {"family": ["slate"], "mode": ["system"], "swatchMode": "dark", "checked": "falsefalsetrue", "attrs": {"data-theme": "slate", "data-mode": "dark"}, "hint": "Follows the system setting (currently dark)."}


def test_custom_css_is_debounced_then_applied_and_flushed_on_close() -> None:
    data = run("""
    controller.show();
    const style = documentTarget.getElementById("codoxearCustomCss");
    textarea.value = ".msg {";
    textarea.emit("input");
    textarea.value = ".msg { color: red; }";
    textarea.emit("input");
    const pending = { timers: timers.length, css: style.textContent };
    runTimers();
    const applied = { css: style.textContent, stored: storage["codoxear.ui.customCss"] };
    textarea.value = "body { margin: 0; }";
    textarea.emit("input");
    controller.hide();
    const flushed = { css: style.textContent, timers: timers.length, stored: storage["codoxear.ui.customCss"] };
    return { pending, applied, flushed };
    """)
    assert data["pending"] == {"timers": 1, "css": ""}
    assert data["applied"] == {"css": ".msg { color: red; }", "stored": ".msg { color: red; }"}
    assert data["flushed"] == {"css": "body { margin: 0; }", "timers": 0, "stored": "body { margin: 0; }"}


def test_appearance_returns_to_defaults_through_the_same_controls() -> None:
    data = run("""
    controller.show();
    family("clay").emit("click");
    mode("light").emit("click");
    textarea.value = "a {}";
    textarea.emit("input");
    runTimers();
    family("paper").emit("click");
    mode("system").emit("click");
    textarea.value = "";
    textarea.emit("input");
    runTimers();
    return { selection: selection(), textarea: textarea.value, attrs: { ...html.attrs }, storage: { ...storage }, state: themeController.get() };
    """)
    assert data["selection"] == {"family": ["paper"], "mode": ["system"], "swatchMode": "light", "checked": "truefalsefalse"}
    assert data["textarea"] == ""
    assert data["attrs"] == {"data-theme": "paper", "data-mode": "light"}
    assert data["storage"] == {}
    assert data["state"]["customCss"] == ""


def test_voice_section_lifecycle_follows_every_show_and_hide_path() -> None:
    data = run("""
    controller.show();
    viewer.emit("click", { target: viewer });  // backdrop-equivalent click closes
    controller.show();
    controller.hide();  // programmatic close (voice Save/Cancel path)
    return { voiceCalls, open: controller.isOpen(), voiceStillMounted: voiceSection.parentNode !== null };
    """)
    assert data["voiceCalls"] == ["activate", "deactivate", "activate", "deactivate"]
    assert data["open"] is False
    assert data["voiceStillMounted"] is True


def test_dispose_stops_rendering_controller_changes() -> None:
    data = run("""
    controller.dispose();
    themeController.applyTheme({ family: "clay" });
    return selection();
    """)
    assert data["family"] == ["paper"]
