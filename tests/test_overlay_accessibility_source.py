"""Modal overlay accessibility: background isolation and focus restore.

Runs the real app_modal.js helpers against the real Settings dialog
(app_settings.js + app_theme.js) in Node's vm: opening a modal must mark the
app root inert/aria-hidden, the close button must receive focus once the
dialog is open, and closing must return focus to the opener after the dialog
is really closed.
"""

from frontend_module_loader import module_path
import json
import subprocess
import textwrap
import unittest


MODAL = module_path("app_modal.js")
THEME = module_path("app_theme.js")
SETTINGS = module_path("app_settings.js")


def run_modal_behavior() -> dict:
    scripts = [path.read_text(encoding="utf-8") for path in (MODAL, THEME, SETTINGS)]
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const calls = [];
        class ElementStub {
          constructor(tag) {
            this.tagName = String(tag).toUpperCase(); this.attrs = {}; this.children = []; this.parentNode = null;
            this.style = {}; this.listeners = {}; this.open = false; this.disabled = false; this.isConnected = true; this.textContent = ""; this.innerHTML = ""; this.value = "";
            const classes = new Set();
            this.classList = { add: (...n) => n.forEach((c) => classes.add(c)), remove: (...n) => n.forEach((c) => classes.delete(c)), toggle: (c, f) => { if (f === undefined ? !classes.has(c) : f) classes.add(c); else classes.delete(c); }, contains: (c) => classes.has(c) };
            Object.defineProperty(this, "className", { get: () => [...classes].join(" "), set: (t) => { classes.clear(); String(t).split(/\\s+/).filter(Boolean).forEach((c) => classes.add(c)); } });
          }
          get id() { return this.attrs.id || ""; } set id(v) { this.attrs.id = String(v); }
          setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; } removeAttribute(k) { delete this.attrs[k]; }
          appendChild(c) { c.parentNode = this; this.children.push(c); return c; } removeChild(c) { this.children = this.children.filter((x) => x !== c); c.parentNode = null; return c; }
          insertBefore(c, ref) { const i = ref ? this.children.indexOf(ref) : -1; if (i < 0) return this.appendChild(c); c.parentNode = this; this.children.splice(i, 0, c); return c; }
          addEventListener(t, h) { (this.listeners[t] ||= []).push(h); }
          emit(t, e = {}) { const ev = { type: t, target: this, currentTarget: this, preventDefault() {}, stopPropagation() {}, ...e }; for (const h of this.listeners[t] || []) h(ev); }
          showModal() { this.open = true; calls.push("showModal"); } close() { this.open = false; calls.push("close"); }
          focus() { calls.push(`focus:${this.attrs.id || this.tagName}`); }
          find(p) { for (const c of this.children) { if (p(c)) return c; const hit = c.find(p); if (hit) return hit; } return null; }
        }
        function el(tag, attrs = {}, children = []) {
          const n = new ElementStub(tag);
          for (const [k, v] of Object.entries(attrs)) { if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else if (k === "html") n.innerHTML = v; else n.setAttribute(k, v); }
          for (const c of children) n.appendChild(c);
          return n;
        }
        const html = new ElementStub("html"); const head = new ElementStub("head"); html.appendChild(head);
        const opener = el("button", { id: "settingsBtnSide" });
        const documentTarget = { documentElement: html, head, activeElement: opener, createElement: (t) => new ElementStub(t), querySelector: () => null, getElementById: (id) => html.find((n) => n.attrs.id === id), contains: () => true };
        // Frames are queued so the test can observe ordering: the close button
        // is focused only once the dialog is open; the opener only once closed.
        const frames = [];
        const ctx = { window: {}, requestAnimationFrame: (fn) => frames.push(fn) };
        vm.createContext(ctx);
        for (const code of __SCRIPTS__) vm.runInContext(code, ctx);
        const { CodoxearModal, CodoxearTheme, CodoxearSettings } = ctx.window;
        const themeController = CodoxearTheme.createThemeController({ documentTarget, storageGetItem: () => null, storageSetItem() {}, storageRemoveItem() {}, matchMedia: () => ({ matches: false, addEventListener() {} }), versionedAssetPath: (p) => p });
        const root = new ElementStub("div");
        const app = { attrs: {}, toggleAttribute(k, on) { if (on) this.attrs[k] = ""; else delete this.attrs[k]; }, setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; } };
        const controller = CodoxearSettings.createSettingsDialogController({
          root, el, iconSvg: () => "", themeController, openButton: opener,
          voiceSection: el("section", { id: "voiceSettingsSection" }), activateVoiceSection() {}, deactivateVoiceSection() {},
          documentTarget, ElementCtor: ElementStub, prepareModalOpen: () => calls.push("prepare"),
          afterModalVisibilityChanged: () => calls.push(`isolation:${CodoxearModal.syncModalIsolation(app, [controller.viewer])}`),
          addEvent: (t, ty, h) => t.addEventListener(ty, h), setTimeout: () => 1, clearTimeout() {},
        });
        const runFrames = () => { while (frames.length) frames.shift()(); };
        opener.emit("click");
        runFrames();
        const shown = { open: controller.isOpen(), attrs: { ...app.attrs }, calls: calls.slice() };
        calls.length = 0;
        controller.viewer.find((n) => n.attrs.id === "settingsCloseBtn").emit("click");
        const beforeFrame = calls.slice();
        runFrames();
        process.stdout.write(JSON.stringify({ shown, hidden: { open: controller.isOpen(), attrs: { ...app.attrs }, beforeFrame, calls: calls.slice() } }));
        """
    ).replace("__SCRIPTS__", json.dumps(scripts))
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


class TestOverlayAccessibilityBehavior(unittest.TestCase):
    def test_modal_show_hide_isolates_background_and_restores_focus(self) -> None:
        result = run_modal_behavior()
        shown = result["shown"]
        self.assertTrue(shown["open"])
        self.assertEqual(shown["attrs"], {"inert": "", "aria-hidden": "true"})
        self.assertEqual(shown["calls"], ["prepare", "showModal", "isolation:true", "focus:settingsViewer"])
        hidden = result["hidden"]
        self.assertFalse(hidden["open"])
        self.assertEqual(hidden["attrs"], {})
        self.assertEqual(hidden["beforeFrame"], ["close", "isolation:false"])
        self.assertEqual(hidden["calls"], ["close", "isolation:false", "focus:settingsBtnSide"])


if __name__ == "__main__":
    unittest.main()
