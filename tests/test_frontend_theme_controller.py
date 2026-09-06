"""app_theme.js is the sole writer of the theme surface.

Executes the controller in Node's vm with a minimal DOM double and asserts
the observable contract: html attributes, the versioned theme <link>, the
custom-CSS <style>, <meta name="theme-color">, localStorage keys, system-mode
resolution through matchMedia, and live re-resolution on scheme change.
"""

from frontend_module_loader import module_path
import json
import os
import subprocess
import textwrap


APP_THEME_JS = module_path("app_theme.js")

DOM_DOUBLE = """
class ElementStub {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.attrs = {};
    this.children = [];
    this.parentNode = null;
    this.textContent = "";
    this.listeners = {};
    this.style = {};
  }
  get id() { return this.attrs.id || ""; }
  set id(value) { if (value) this.attrs.id = String(value); else delete this.attrs.id; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attrs, name) ? this.attrs[name] : null; }
  removeAttribute(name) { delete this.attrs[name]; }
  appendChild(child) { return this.insertBefore(child, null); }
  insertBefore(child, reference) {
    if (child.parentNode) child.parentNode.removeChild(child);
    const index = reference ? this.children.indexOf(reference) : -1;
    if (index < 0) this.children.push(child); else this.children.splice(index, 0, child);
    child.parentNode = this;
    return child;
  }
  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    child.parentNode = null;
    return child;
  }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  emit(type) { for (const handler of this.listeners[type] || []) handler({ type }); }
}
function makeDocument() {
  const html = new ElementStub("html");
  const head = new ElementStub("head");
  html.appendChild(head);
  const documentTarget = {
    documentElement: html,
    head,
    createElement: (tag) => new ElementStub(tag),
    querySelector(selector) {
      if (selector === 'meta[name="theme-color"]') return head.children.find((node) => node.tagName === "META" && node.attrs.name === "theme-color") || null;
      return null;
    },
    getElementById(id) {
      const walk = (node) => {
        if (node.attrs && node.attrs.id === id) return node;
        for (const child of node.children || []) { const hit = walk(child); if (hit) return hit; }
        return null;
      };
      return walk(html);
    },
  };
  return { html, head, documentTarget };
}
function makeStorage(initial = {}) {
  const data = { ...initial };
  return {
    data,
    get: (key) => (Object.prototype.hasOwnProperty.call(data, key) ? data[key] : null),
    set: (key, value) => { data[key] = String(value); },
    remove: (key) => { delete data[key]; },
  };
}
function makeMatchMedia(matches) {
  const query = { matches, listeners: [], addEventListener(type, handler) { this.listeners.push(handler); }, removeEventListener(type, handler) { this.listeners = this.listeners.filter((h) => h !== handler); } };
  return { query, matchMedia: (text) => (text === "(prefers-color-scheme: dark)" ? query : null) };
}
function headSummary(head) {
  return head.children.map((node) => `${node.tagName.toLowerCase()}${node.attrs.id ? "#" + node.attrs.id : ""}${node.attrs.href ? "[" + node.attrs.href + "]" : ""}${node.attrs.name ? "[" + node.attrs.name + "=" + node.attrs.content + "]" : ""}`);
}
"""


def run(program_body: str) -> dict:
    source = APP_THEME_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const ctx = { window: {} };
        vm.createContext(ctx);
        vm.runInContext(__SOURCE__, ctx);
        const CodoxearTheme = ctx.window.CodoxearTheme;
        __DOM__
        const out = (function () { __BODY__ })();
        process.stdout.write(JSON.stringify(out));
        """
    ).replace("__SOURCE__", json.dumps(source)).replace("__DOM__", DOM_DOUBLE).replace("__BODY__", program_body)
    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(result.stdout)


def make_controller_snippet(storage_initial: str = "{}", dark: str = "false") -> str:
    return f"""
    const {{ html, head, documentTarget }} = makeDocument();
    const storage = makeStorage({storage_initial});
    const {{ query, matchMedia }} = makeMatchMedia({dark});
    const controller = CodoxearTheme.createThemeController({{
      documentTarget,
      storageGetItem: storage.get,
      storageSetItem: storage.set,
      storageRemoveItem: storage.remove,
      matchMedia,
      versionedAssetPath: (path) => `${{path}}?v=test1`,
    }});
    """


def test_defaults_render_paper_following_a_light_system() -> None:
    data = run(make_controller_snippet() + """
    return { attrs: html.attrs, head: headSummary(head), state: controller.get(), storage: storage.data };
    """)
    assert data["attrs"] == {"data-theme": "paper", "data-mode": "light"}
    assert data["head"] == ["meta[theme-color=#ffffff]", "link#codoxearThemeLink[themes/paper.css?v=test1]", "style#codoxearCustomCss"]
    assert data["state"] == {"family": "paper", "mode": "system", "resolvedMode": "light", "customCss": ""}
    assert data["storage"] == {}


def test_apply_theme_writes_attributes_link_meta_and_storage() -> None:
    data = run(make_controller_snippet() + """
    const notified = [];
    controller.subscribe((theme) => notified.push(`${theme.family}/${theme.mode}/${theme.resolvedMode}`));
    controller.applyTheme({ family: "slate", mode: "dark" });
    const pending = headSummary(head);
    const links = head.children.filter((node) => node.tagName === "LINK");
    links[links.length - 1].emit("load");
    return { attrs: html.attrs, pending, head: headSummary(head), storage: storage.data, notified, meta: head.children[0].attrs.content };
    """)
    assert data["attrs"] == {"data-theme": "slate", "data-mode": "dark"}
    # Swap-on-load: the paper link stays until slate has loaded, then retires.
    assert data["pending"] == ["meta[theme-color=#171717]", "link[themes/paper.css?v=test1]", "link[themes/slate.css?v=test1]", "style#codoxearCustomCss"]
    assert data["head"] == ["meta[theme-color=#171717]", "link#codoxearThemeLink[themes/slate.css?v=test1]", "style#codoxearCustomCss"]
    assert data["storage"] == {"codoxear.ui.theme.family": "slate", "codoxear.ui.theme.mode": "dark"}
    assert data["notified"] == ["paper/system/light", "slate/dark/dark"]


def test_persisted_state_and_dark_system_resolve_at_construction() -> None:
    data = run(make_controller_snippet('{ "codoxear.ui.theme.family": "clay", "codoxear.ui.customCss": ".msg { color: red; }" }', "true") + """
    const style = documentTarget.getElementById("codoxearCustomCss");
    return { attrs: html.attrs, state: controller.get(), css: style.textContent, meta: head.children[0].attrs.content };
    """)
    assert data["attrs"] == {"data-theme": "clay", "data-mode": "dark"}
    assert data["state"] == {"family": "clay", "mode": "system", "resolvedMode": "dark", "customCss": ".msg { color: red; }"}
    assert data["css"] == ".msg { color: red; }"
    assert data["meta"] == "#161310"


def test_system_mode_re_resolves_on_scheme_change_and_explicit_modes_ignore_it() -> None:
    data = run(make_controller_snippet() + """
    const modes = [];
    controller.subscribe((theme) => modes.push(theme.resolvedMode));
    query.matches = true;
    for (const handler of query.listeners) handler({ matches: true });
    const afterSystemFlip = { ...html.attrs };
    controller.applyTheme({ mode: "light" });
    query.matches = false;
    for (const handler of query.listeners) handler({ matches: false });
    query.matches = true;
    for (const handler of query.listeners) handler({ matches: true });
    const afterExplicit = { ...html.attrs };
    controller.dispose();
    return { afterSystemFlip, afterExplicit, modes, listenersAfterDispose: query.listeners.length };
    """)
    assert data["afterSystemFlip"] == {"data-theme": "paper", "data-mode": "dark"}
    assert data["afterExplicit"] == {"data-theme": "paper", "data-mode": "light"}
    assert data["modes"] == ["light", "dark", "light"]
    assert data["listenersAfterDispose"] == 0


def test_custom_css_lands_in_the_style_element_and_persists() -> None:
    data = run(make_controller_snippet() + """
    controller.setCustomCss(":root { --accent: hotpink; }");
    const style = documentTarget.getElementById("codoxearCustomCss");
    const css = style.textContent;
    const stored = { ...storage.data };
    controller.setCustomCss("");
    return { css, stored, cleared: storage.data, order: headSummary(head) };
    """)
    assert data["css"] == ":root { --accent: hotpink; }"
    assert data["stored"] == {"codoxear.ui.customCss": ":root { --accent: hotpink; }"}
    assert data["cleared"] == {}
    assert data["order"][-1] == "style#codoxearCustomCss"


def test_invalid_persisted_values_fall_back_to_paper_and_system() -> None:
    data = run(make_controller_snippet('{ "codoxear.ui.theme.family": "neon", "codoxear.ui.theme.mode": "sepia" }') + """
    const applied = controller.applyTheme({ family: "Solar", mode: "DARK" });
    return { state: controller.get(), applied, attrs: html.attrs, storage: storage.data };
    """)
    assert data["applied"]["family"] == "paper"
    assert data["applied"]["mode"] == "dark"
    assert data["attrs"] == {"data-theme": "paper", "data-mode": "dark"}
    assert data["storage"] == {"codoxear.ui.theme.mode": "dark"}


def test_controller_adopts_boot_rendered_nodes_without_refetching() -> None:
    # index.html renders the link/style before the module loads; the
    # controller must adopt those nodes rather than duplicate them.
    data = run("""
    const { html, head, documentTarget } = makeDocument();
    const meta = documentTarget.createElement("meta"); meta.setAttribute("name", "theme-color"); meta.setAttribute("content", "#ffffff"); head.appendChild(meta);
    const bootLink = documentTarget.createElement("link"); bootLink.id = "codoxearThemeLink"; bootLink.setAttribute("rel", "stylesheet"); bootLink.setAttribute("href", "themes/clay.css?v=test1"); head.appendChild(bootLink);
    const bootStyle = documentTarget.createElement("style"); bootStyle.id = "codoxearCustomCss"; bootStyle.textContent = "a{}"; head.appendChild(bootStyle);
    const storage = makeStorage({ "codoxear.ui.theme.family": "clay", "codoxear.ui.customCss": "a{}" });
    const { matchMedia } = makeMatchMedia(false);
    const controller = CodoxearTheme.createThemeController({
      documentTarget, storageGetItem: storage.get, storageSetItem: storage.set, storageRemoveItem: storage.remove, matchMedia,
      versionedAssetPath: (path) => `${path}?v=test1`,
    });
    return { head: headSummary(head), sameLink: documentTarget.getElementById("codoxearThemeLink") === bootLink, meta: meta.attrs.content };
    """)
    assert data["head"] == ["meta[theme-color=#faf7f0]", "link#codoxearThemeLink[themes/clay.css?v=test1]", "style#codoxearCustomCss"]
    assert data["sameLink"] is True
