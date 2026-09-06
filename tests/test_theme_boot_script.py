"""The inline theme boot script projects the stored theme before the module loads.

index.html carries a render-blocking inline script after the app.css link
that reads the persisted family/mode from localStorage, resolves "system"
through matchMedia, and writes the html attributes, the theme stylesheet
link, the custom-CSS style, and <meta name="theme-color">. iOS Safari applies
theme-color on navigation, so the boot projection — not app_theme.js's later
render — decides the status-bar color the user sees after a reload.

The script is inline and cannot import, so it repeats app_theme.js's
THEME_COLORS table. These tests execute the real script against a DOM double
for every family x mode and compare the written color with the module's
table, so the two copies cannot drift.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import textwrap

from frontend_module_loader import module_path
import pytest

INDEX_HTML = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "index.html"
APP_THEME_JS = module_path("app_theme.js")


class _HeadScripts(HTMLParser):
    """Collects <head> children in order: ("link", attrs) and ("script", text)."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[tuple[str, object]] = []
        self._script: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "link":
            self.nodes.append(("link", dict(attrs)))
        elif tag == "script":
            self._script = []

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.nodes.append(("script", "".join(self._script)))
            self._script = None


def theme_boot_script() -> str:
    parser = _HeadScripts()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    for index, (kind, payload) in enumerate(parser.nodes):
        if kind == "link" and str(payload.get("href", "")).startswith("app.css"):
            kind_next, script = parser.nodes[index + 1]
            assert kind_next == "script", "the theme boot script must directly follow the app.css link"
            return script.replace("__CODOXEAR_ASSET_VERSION__", "v-test")
    raise AssertionError("app.css link not found in index.html head")


def run_boot(family: str | None, mode: str | None, system_dark: bool) -> dict:
    program = textwrap.dedent(
        """
        const vm = require("vm");
        const storage = __STORAGE__;
        const attrs = {};
        const meta = { attrs: { name: "theme-color", content: "#ffffff" }, setAttribute(k, v) { this.attrs[k] = String(v); } };
        const written = [];
        const ctx = {
          window: {
            localStorage: { getItem: (key) => (key in storage ? storage[key] : null) },
            matchMedia: (query) => ({ media: query, matches: __SYSTEM_DARK__ }),
          },
          document: {
            documentElement: { setAttribute(k, v) { attrs[k] = String(v); } },
            querySelector: (selector) => (selector === 'meta[name="theme-color"]' ? meta : null),
            write: (html) => written.push(html),
          },
        };
        vm.createContext(ctx);
        vm.runInContext(__BOOT__, ctx);
        vm.runInContext(__THEME__, ctx);
        process.stdout.write(JSON.stringify({ attrs, meta: meta.attrs.content, written, table: ctx.window.CodoxearTheme.THEME_COLORS }));
        """
    )
    storage = {}
    if family is not None:
        storage["codoxear.ui.theme.family"] = family
    if mode is not None:
        storage["codoxear.ui.theme.mode"] = mode
    program = (
        program.replace("__STORAGE__", json.dumps(storage))
        .replace("__SYSTEM_DARK__", "true" if system_dark else "false")
        .replace("__BOOT__", json.dumps(theme_boot_script()))
        .replace("__THEME__", json.dumps(APP_THEME_JS.read_text(encoding="utf-8")))
    )
    result = subprocess.run(["node", "-e", program], check=True, capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
    return json.loads(result.stdout)


@pytest.mark.parametrize("family", ["paper", "clay", "slate"])
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_boot_writes_the_module_theme_color_for_every_stored_variant(family: str, mode: str) -> None:
    data = run_boot(family, mode, system_dark=False)
    assert data["attrs"] == {"data-theme": family, "data-mode": mode}
    assert data["meta"] == data["table"][family][mode]
    assert any(f"themes/{family}.css?v=v-test" in chunk for chunk in data["written"])


def test_boot_table_covers_exactly_the_module_table() -> None:
    colors = {(family, mode): run_boot(family, mode, system_dark=False)["meta"] for family in ("paper", "clay", "slate") for mode in ("light", "dark")}
    table = run_boot(None, None, system_dark=False)["table"]
    assert colors == {(family, mode): table[family][mode] for family in table for mode in table[family]}


@pytest.mark.parametrize("system_dark,expected_mode", [(False, "light"), (True, "dark")])
def test_boot_resolves_system_mode_before_choosing_the_color(system_dark: bool, expected_mode: str) -> None:
    data = run_boot("slate", None, system_dark=system_dark)
    assert data["attrs"] == {"data-theme": "slate", "data-mode": expected_mode}
    assert data["meta"] == data["table"]["slate"][expected_mode]


def test_boot_defaults_unknown_storage_to_paper() -> None:
    data = run_boot("neon", "sepia", system_dark=False)
    assert data["attrs"] == {"data-theme": "paper", "data-mode": "light"}
    assert data["meta"] == data["table"]["paper"]["light"]
