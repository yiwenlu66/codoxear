"""Behavioral regression coverage for the file-viewer save-button geometry.

The save control (``#fileEditBtn``) is compact 32px header chrome.  It once
also carried ``.primary`` in its save state, which pulled in the global
``button.primary { min-height: var(--ctl) !important; }`` full-size-primary
clamp and rendered the button 44px tall next to 32px siblings.  These tests
execute the real shell DOM builder and the real ``updateFileEditButton()``
state machine, then evaluate the stylesheet cascade for the mounted element,
so the contract is verified as computed geometry rather than source text.
"""

from __future__ import annotations

from frontend_module_loader import module_path

import json
from pathlib import Path
import re
import subprocess
import textwrap

from tinycss2 import parse_declaration_list, parse_rule_list, parse_stylesheet, serialize


ROOT = Path(__file__).resolve().parents[1]
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
APP_SHELL_JS = module_path("app_shell.js")
APP_FILE_VIEWER_OPERATIONS_JS = module_path("app_file_viewer_operations.js")


# ---------------------------------------------------------------------------
# Real DOM mounting (node VM)
# ---------------------------------------------------------------------------

def _run_node(js: str) -> dict:
    result = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _mounted_header_actions() -> dict[str, str]:
    """Mount the real shell DOM and report id -> class for header action buttons."""
    source = APP_SHELL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        class Node {{
          constructor(tag, attrs = {{}}) {{
            this.tag = tag;
            this.attrs = attrs;
            this.className = attrs.class || "";
            this.id = attrs.id || "";
            this.children = [];
            this.parent = null;
            this.style = {{}};
          }}
          appendChild(child) {{ child.parent = this; this.children.push(child); return child; }}
          append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
          set innerHTML(_value) {{ this.children = []; }}
        }}
        const el = (tag, attrs = {{}}, children = []) => {{
          const node = new Node(tag, attrs);
          (Array.isArray(children) ? children : [children]).forEach((child) => child && node.appendChild(child));
          return node;
        }};
        const ctx = {{ window: {{}}, document: {{}} }};
        ctx.window.CodoxearQueue = {{ createQueueDom: () => ({{}}) }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const root = new Node("root");
        const shell = ctx.window.CodoxearShell.createShellDOM({{
          root,
          el,
          iconSvg: (name) => `<${{name}}>`,
          resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
        }});
        const voiceStub = {{ createVoiceDom: () => ({{}}) }};
        const modal = ctx.window.CodoxearShell.createApplicationModalDOM({{
          root,
          el,
          iconSvg: (name) => `<${{name}}>`,
          windowTarget: ctx.window,
          codoxearVoice: voiceStub,
          voiceHost: shell.elements.voiceHost,
        }});
        const actions = modal.fileEditBtn.parent;
        const shape = {{}};
        actions.children.forEach((child) => {{ shape[child.id] = child.className; }});
        process.stdout.write(JSON.stringify(shape));
        """
    )
    return _run_node(js)


def _update_file_edit_button_states() -> dict[str, dict]:
    """Drive the real updateFileEditButton() through its save-state transitions."""
    source = APP_FILE_VIEWER_OPERATIONS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);

        function fakeButton() {{
          const classes = new Set(["icon-btn"]);
          return {{
            disabled: false,
            title: "",
            innerHTML: "",
            attrs: {{}},
            classList: {{
              toggle(name, force) {{ if (force) classes.add(name); else classes.delete(name); }},
              contains(name) {{ return classes.has(name); }},
            }},
            setAttribute(name, value) {{ this.attrs[name] = value; }},
            snapshot() {{ return {{
              classes: Array.from(classes).sort(),
              disabled: this.disabled,
              ariaLabel: this.attrs["aria-label"] || "",
              icon: this.innerHTML,
            }}; }},
          }};
        }}

        const button = fakeButton();
        const state = {{ editMode: false, savePending: false }};
        const noop = () => {{}};
        const falsy = () => false;
        const runtime = ctx.window.CodoxearFileViewerOperations.createFileViewerOperationsRuntime(new Proxy({{
          el: (tag) => ({{ tag }}),
          fileEditButton: button,
          fileStatus: {{ textContent: "" }},
          iconSvg: (name) => name,
          currentFileEditMode: () => state.editMode,
          currentFileEditorState: () => ({{}}),
          focusEditor: () => null,
          activeFileCanEnterEditMode: () => true,
          isUnavailable: falsy,
        }}, {{ get: (target, prop) => (prop in target ? target[prop] : noop) }}));

        const out = {{}};
        runtime.updateFileEditButton();
        out.readOnly = button.snapshot();
        state.editMode = true;
        runtime.updateFileEditButton();
        out.editMode = button.snapshot();
        runtime.markActiveFileSavePending({{ path: "f.py", apiPath: "/api/file", gitPath: false, token: 1 }});
        runtime.updateFileEditButton();
        out.savePending = button.snapshot();
        process.stdout.write(JSON.stringify(out));
        """
    )
    return _run_node(js)


# ---------------------------------------------------------------------------
# Stylesheet cascade evaluation (tinycss2)
# ---------------------------------------------------------------------------

def _rules_at_width(css: str, width: int) -> list[tuple[str, dict[str, str], int]]:
    def media_applies(prelude: str) -> bool:
        max_widths = [int(value) for value in re.findall(r"max-width\s*:\s*(\d+)px", prelude)]
        min_widths = [int(value) for value in re.findall(r"min-width\s*:\s*(\d+)px", prelude)]
        return all(width <= value for value in max_widths) and all(width >= value for value in min_widths)

    def declarations(tokens: list[object]) -> dict[str, str]:
        return {
            declaration.lower_name: serialize(declaration.value).strip()
            for declaration in parse_declaration_list(tokens, skip_comments=True, skip_whitespace=True)
            if declaration.type == "declaration"
        }

    def walk(rules: list[object], active: bool = True) -> list[tuple[str, dict[str, str], int]]:
        computed: list[tuple[str, dict[str, str], int]] = []
        for order, rule in enumerate(rules):
            if rule.type == "at-rule" and rule.at_keyword == "media" and rule.content is not None:
                computed.extend(walk(
                    parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True),
                    active and media_applies(serialize(rule.prelude)),
                ))
            elif active and rule.type == "qualified-rule":
                declaration_map = declarations(rule.content)
                for selector in serialize(rule.prelude).split(","):
                    computed.append((selector.strip(), declaration_map, order))
        return computed

    return walk(parse_stylesheet(css, skip_comments=True, skip_whitespace=True))


class _Element:
    """Minimal element for selector matching: tag, id, classes, ancestors."""

    def __init__(self, tag: str, id: str = "", classes: tuple[str, ...] = (),
                 disabled: bool = False, ancestors: tuple["_Element", ...] = ()) -> None:
        self.tag = tag
        self.id = id
        self.classes = set(classes)
        self.disabled = disabled
        self.ancestors = ancestors


def _matches_compound(element: _Element, compound: str) -> bool:
    if ":root" in compound:
        return element.tag == "html"
    for token in re.findall(r"[#.]?[\w-]+|:(?:disabled|hover)", compound):
        if token.startswith("#"):
            if element.id != token[1:]:
                return False
        elif token.startswith("."):
            if token[1:] not in element.classes:
                return False
        elif token == ":disabled":
            if not element.disabled:
                return False
        elif token == ":hover":
            return False
        elif element.tag != token:
            return False
    return True


def _selector_specificity(selector: str) -> tuple[int, int, int]:
    ids = len(re.findall(r"#[\w-]+", selector))
    classes = len(re.findall(r"\.[\w-]+", selector)) + len(re.findall(r":[\w-]+", selector))
    types = len(re.findall(r"(?<![#\w.-])[a-zA-Z][\w-]*(?![\w-]*\()", selector))
    return (ids, classes, types)


def _matches_selector(element: _Element, selector: str) -> bool:
    if ":not(" in selector or "::" in selector or "[" in selector:
        return False
    compounds = selector.split()
    if not compounds or not _matches_compound(element, compounds[-1]):
        return False
    ancestor_index = 0
    for compound in reversed(compounds[:-1]):
        found = False
        while ancestor_index < len(element.ancestors):
            ancestor = element.ancestors[ancestor_index]
            ancestor_index += 1
            if _matches_compound(ancestor, compound):
                found = True
                break
        if not found:
            return False
    return True


def _computed_declarations(element: _Element, width: int) -> dict[str, str]:
    matched: list[tuple[tuple[int, int, int], int, dict[str, str]]] = []
    rules = _rules_at_width(APP_CSS.read_text(encoding="utf-8"), width)
    for selector, declarations, order in rules:
        if _matches_selector(element, selector):
            matched.append((_selector_specificity(selector), order, declarations))
    matched.sort(key=lambda item: (item[0], item[1]))
    computed: dict[str, str] = {}
    for _, _, declarations in matched:
        computed.update(declarations)
    return computed


def _resolve_px(value: str, variables: dict[str, str]) -> int:
    resolved = value
    for name, variable_value in variables.items():
        resolved = resolved.replace(f"var({name})", variable_value)
    numbers = [int(number) for number in re.findall(r"(?<![-\w])(\d+)px", resolved)]
    if len(numbers) == 1:
        return numbers[0]
    raise AssertionError(f"cannot resolve CSS length: {value!r} -> {resolved!r}")


def _root_variables(width: int) -> dict[str, str]:
    return _computed_declarations(_Element("html"), width)


def _file_edit_button(**overrides) -> _Element:
    actions = _Element("div", classes=("actions",))
    header = _Element("div", classes=("fileViewerHeader",), ancestors=(actions,))
    actions.ancestors = (header, _Element("div", classes=("fileViewer",)))
    params = dict(tag="button", id="fileEditBtn", classes=("icon-btn", "active"))
    params.update(overrides)
    return _Element(ancestors=(actions, header), **params)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_button_shares_compact_chrome_classes_with_siblings() -> None:
    actions = _mounted_header_actions()
    assert actions["fileEditBtn"] == "icon-btn"
    for sibling in ("fileModeDiffBtn", "fileModePreviewBtn", "fileDownloadBtn", "fileCloseBtn"):
        assert actions[sibling] == "icon-btn"


def test_update_file_edit_button_uses_active_not_primary() -> None:
    states = _update_file_edit_button_states()
    assert states["readOnly"]["classes"] == ["icon-btn"]
    assert states["editMode"]["classes"] == ["active", "icon-btn"]
    assert states["editMode"]["ariaLabel"] == "Save file"
    assert states["editMode"]["icon"] == "save"
    # Save-pending stays visually inverted but disabled; still no .primary.
    assert "primary" not in states["savePending"]["classes"]
    assert "active" in states["savePending"]["classes"]
    assert states["savePending"]["disabled"] is True


def test_save_button_computes_compact_32px_geometry() -> None:
    variables = _root_variables(393)
    edit_classes = tuple(_update_file_edit_button_states()["editMode"]["classes"])
    computed = _computed_declarations(_file_edit_button(classes=edit_classes), 393)
    assert _resolve_px(computed["height"], variables) == 32
    assert _resolve_px(computed["min-height"], variables) == 32
    assert _resolve_px(computed["width"], variables) == 32
    # The inverted save state is the ink-on-paper primary treatment.
    assert computed["background"] == "var(--ink)"
    assert computed["color"] == "var(--paper)"


def test_save_button_disabled_save_state_stays_visibly_disabled() -> None:
    pending_classes = tuple(_update_file_edit_button_states()["savePending"]["classes"])
    computed = _computed_declarations(_file_edit_button(classes=pending_classes, disabled=True), 393)
    assert computed["background"] == "var(--ink-disabled)"
    assert computed["cursor"] == "not-allowed"
