"""Behavioral regression coverage for the iOS composer layout contract.

The VM fixture mounts the real composer DOM from ``app_shell.js``.  The small CSS
cascade evaluator then computes the declarations that apply to that mounted
textarea at desktop and iPhone widths, resolving type/spacing variables and a
concrete safe-area inset.  This tests the user-visible geometry rather than
matching stylesheet text.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
APP_SHELL_JS = ROOT / "codoxear" / "static" / "app_shell.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


class _ViewportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.content: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        properties = dict(attrs)
        if properties.get("name") == "viewport":
            self.content = properties.get("content")


def _matching_brace(css: str, opening_brace: int) -> int:
    depth = 1
    for index in range(opening_brace + 1, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("unclosed CSS block")


def _declarations(body: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for declaration in body.split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", 1)
        name, value = name.strip(), value.strip()
        if name and value:
            declarations[name] = value
    return declarations


def _rules_at_width(css: str, width: int) -> list[tuple[str, dict[str, str]]]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)

    def walk(block: str, active: bool = True) -> list[tuple[str, dict[str, str]]]:
        rules: list[tuple[str, dict[str, str]]] = []
        cursor = 0
        while cursor < len(block):
            opening_brace = block.find("{", cursor)
            if opening_brace < 0:
                break
            header = block[cursor:opening_brace].strip()
            closing_brace = _matching_brace(block, opening_brace)
            body = block[opening_brace + 1 : closing_brace]
            cursor = closing_brace + 1
            if header.startswith("@media"):
                max_widths = [int(value) for value in re.findall(r"max-width\s*:\s*(\d+)px", header)]
                min_widths = [int(value) for value in re.findall(r"min-width\s*:\s*(\d+)px", header)]
                applies = all(width <= value for value in max_widths) and all(width >= value for value in min_widths)
                rules.extend(walk(body, active and applies))
            elif active and not header.startswith("@"):
                rules.extend((selector.strip(), _declarations(body)) for selector in header.split(","))
        return rules

    return walk(css)


def _computed_rule(width: int, selector: str) -> dict[str, str]:
    computed: dict[str, str] = {}
    for candidate, declarations in _rules_at_width(APP_CSS.read_text(encoding="utf-8"), width):
        if candidate == selector:
            computed.update(declarations)
    if not computed:
        raise AssertionError(f"missing CSS rule for {selector}")
    return computed


def _resolve_px(value: str, variables: dict[str, str], safe_area_bottom: int = 0) -> int:
    resolved = value
    for name, variable_value in variables.items():
        resolved = resolved.replace(f"var({name})", variable_value)
    resolved = resolved.replace("env(safe-area-inset-bottom)", f"{safe_area_bottom}px")
    numbers = [int(number) for number in re.findall(r"(?<![-\w])(\d+)px", resolved)]
    if resolved.startswith("calc("):
        return sum(numbers)
    if len(numbers) == 1:
        return numbers[0]
    raise AssertionError(f"cannot resolve CSS length: {value!r} -> {resolved!r}")


def _padding_bottom(value: str, variables: dict[str, str], safe_area_bottom: int) -> int:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character.isspace() and depth == 0:
            if start < index:
                parts.append(value[start:index])
            start = index + 1
    if start < len(value):
        parts.append(value[start:])
    if len(parts) in {3, 4}:
        return _resolve_px(parts[2], variables, safe_area_bottom)
    raise AssertionError(f"expected three- or four-value padding shorthand, got {value!r}")


def _mounted_composer_shape() -> dict[str, str | None]:
    source = APP_SHELL_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        class Node {{
          constructor(tag, attrs = {{}}) {{
            this.tag = tag;
            this.attrs = attrs;
            this.className = attrs.class || "";
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
          children.forEach((child) => node.appendChild(child));
          return node;
        }};
        const ctx = {{ window: {{}}, document: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const root = new Node("root");
        const {{ elements }} = ctx.window.CodoxearShell.createShellDOM({{
          root,
          el,
          iconSvg: (name) => `<${{name}}>`,
          resolveAppUrl: (value) => value,
          versionedShellAssetPath: (value) => value,
        }});
        process.stdout.write(JSON.stringify({{
          tag: elements.textarea.tag,
          inputWrap: elements.textarea.parent && elements.textarea.parent.className,
          inputRow: elements.textarea.parent && elements.textarea.parent.parent && elements.textarea.parent.parent.className,
          composer: elements.composer.className,
        }}));
        """
    )
    result = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_ios_composer_textarea_and_safe_area_geometry() -> None:
    assert _mounted_composer_shape() == {
        "tag": "textarea",
        "inputWrap": "inputWrap",
        "inputRow": "composerInputRow",
        "composer": "composer",
    }

    variables = _computed_rule(393, ":root")
    composer_input = _computed_rule(393, ".composer textarea")
    phone_composer = _computed_rule(393, ".composer")

    # iOS Safari only auto-zooms editable controls below 16 CSS px.
    assert _resolve_px(composer_input["font-size"], variables) >= 16
    # A 34px iPhone home-indicator inset must increase the 393px compositor's
    # effective bottom padding by exactly that inset, proving it is part of
    # the computed geometry instead of a fixed replacement gap.
    base_padding = _padding_bottom(phone_composer["padding"], variables, safe_area_bottom=0)
    inset_padding = _padding_bottom(phone_composer["padding"], variables, safe_area_bottom=34)
    assert inset_padding == base_padding + 34


def test_viewport_enables_ios_safe_area_layout() -> None:
    parser = _ViewportParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    assert parser.content is not None
    directives = {part.strip() for part in parser.content.split(",")}
    assert {"width=device-width", "initial-scale=1", "viewport-fit=cover"} <= directives
