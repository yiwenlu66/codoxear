import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


def _extract_ctx_chip_attrs(source: str) -> str:
    match = re.search(r'const ctxChip = el\("button", \{(?P<attrs>.*?)\}\);', source, re.S)
    if not match:
        raise AssertionError("ctxChip is not constructed as a native button")
    return match.group("attrs")


def _extract_block_after(source: str, marker: str) -> str:
    start = source.index(marker)
    end = source.index("\n        };", start)
    return source[start:end]


def _extract_css_rule(source: str, selector: str) -> str:
    pattern = re.compile(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\s*\}}", re.S)
    match = pattern.search(source)
    if not match:
        raise AssertionError(f"missing CSS rule for {selector}")
    return match.group("body")


class TestContextChipAccessibilitySource(unittest.TestCase):
    def test_context_chip_is_native_button_with_stable_action_name(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        attrs = _extract_ctx_chip_attrs(source)

        self.assertNotIn('const ctxChip = el("span"', source)
        self.assertIn('class: "status-chip"', attrs)
        self.assertIn('id: "ctxChip"', attrs)
        self.assertIn('text: ""', attrs)
        self.assertIn('type: "button"', attrs)
        self.assertIn('"aria-label": "Context usage details"', attrs)
        self.assertIn('ctxChip.style.display = "none";', source)
        self.assertIn('ctxChip.disabled = true;', source)

    def test_context_chip_visibility_preserves_token_projection_boundaries(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        set_context = source[source.index("function setContext(tok)") : source.index("ctxChip.onclick", source.index("function setContext(tok)"))]

        self.assertIn('ctxChip.style.display = "none";', set_context)
        self.assertIn('ctxChip.disabled = true;', set_context)
        self.assertIn('ctxChip.textContent = "";', set_context)
        self.assertIn('ctxChip.title = "";', set_context)
        self.assertIn('ctxChip.style.display = "inline-flex";', set_context)
        self.assertIn('ctxChip.disabled = false;', set_context)
        self.assertIn('ctxChip.textContent = p === null ? "Ctx" : `Ctx ${p}%`;', set_context)
        self.assertIn('ctxChip.title = `Context input: ${used}/${lastToken.maxInput} tokens (${lastToken.reserved} reserved; window ${ctx}).`;', set_context)
        self.assertNotIn("context_window =", set_context)
        self.assertNotIn("fetch(", set_context)

    def test_context_chip_native_activation_reuses_existing_click_action(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        click_block = _extract_block_after(source, "ctxChip.onclick = () => {")

        self.assertIn("if (!lastToken) return;", click_block)
        self.assertIn('setToast(`ctx ${lastToken.used}/${lastToken.ctx} (${lastToken.pct ?? "?"}% left)`);', click_block)
        self.assertNotIn("ctxChip.onkeydown", source)
        self.assertNotIn('ctxChip.setAttribute("role"', source)
        self.assertNotIn("ctxChip.tabIndex", source)

    def test_status_chip_button_css_preserves_existing_density(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        status_rule = _extract_css_rule(css, ".status-chip")
        button_rule = _extract_css_rule(css, "button.status-chip")

        for declaration in (
            "display: inline-flex;",
            "align-items: center;",
            "padding: 4px 10px;",
            "border-radius: 999px;",
            "border: 1px solid var(--border);",
            "background: rgba(255, 255, 255, 0.9);",
            "color: var(--muted);",
            "font-size: 12px;",
            "line-height: 1.2;",
            "white-space: nowrap;",
        ):
            self.assertIn(declaration, status_rule)
        for declaration in (
            "appearance: none;",
            "-webkit-appearance: none;",
            "margin: 0;",
            "font-family: inherit;",
            "font-weight: inherit;",
            "cursor: pointer;",
        ):
            self.assertIn(declaration, button_rule)
        self.assertIn(".status-chip {\n          font-size: 11px;\n          padding: 3px 8px;", css)


if __name__ == "__main__":
    unittest.main()
