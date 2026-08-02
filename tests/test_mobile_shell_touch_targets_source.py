import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_CSS = ROOT / "codoxear" / "static" / "app.css"
APP_HINT_MODE = ROOT / "codoxear" / "static" / "app_hint_mode.js"
APP_SHELL = ROOT / "codoxear" / "static" / "app_shell.js"

CHROME_SELECTORS = [
    ".topActions .icon-btn",
    ".pill > .icon-btn",
    ".sidebar header .icon-btn",
    ".chatNavRail .icon-btn",
]

COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
RULE_RE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", re.MULTILINE)


def media_block(css: str, marker: str) -> str:
    start = css.index(marker)
    open_brace = css.index("{", start)
    depth = 1
    pos = open_brace + 1
    while pos < len(css) and depth:
        if css[pos] == "{":
            depth += 1
        elif css[pos] == "}":
            depth -= 1
        pos += 1
    return css[open_brace + 1 : pos - 1]


def selector_bodies(css: str, selector: str) -> list[str]:
    css = COMMENT_RE.sub("", css)
    bodies: list[str] = []
    for match in RULE_RE.finditer(css):
        selectors = [part.strip() for part in match.group("selectors").split(",")]
        if selector in selectors:
            bodies.append(match.group("body"))
    return bodies


def body_with(css: str, selector: str, declaration: str) -> str:
    for body in selector_bodies(css, selector):
        if declaration in body:
            return body
    raise AssertionError(f"missing {declaration!r} in a CSS rule for {selector}")


class TestPaperDesignLanguageSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.css = APP_CSS.read_text(encoding="utf-8")
        cls.touch = media_block(cls.css, "@media (max-width: 700px), (pointer: coarse)")
        cls.phone = media_block(cls.css, "@media (max-width: 520px)")

    def test_root_exposes_paper_palette_and_control_tokens(self) -> None:
        root = selector_bodies(self.css, ":root")[0]
        expected = {
            "--ink": "#141111",
            "--paper": "#ffffff",
            "--bg": "#f6f5f1",
            "--wash": "#efeee9",
            "--hairline": "#dcdad4",
            "--border": "#141111",
            "--muted": "#6b6862",
            "--accent": "#141111",
            "--accent-weak": "#efeee9",
            "--ctl": "38px",
            "--ctl-chrome": "32px",
        }
        for name, value in expected.items():
            self.assertRegex(root, rf"{re.escape(name)}\s*:\s*{re.escape(value)}\s*;")
        self.assertIn(
            "--font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;",
            root,
        )

    def test_primary_controls_use_inversion_and_solid_disabled_state(self) -> None:
        primary = body_with(self.css, "button.primary", "background: var(--ink)")
        self.assertIn("border-color: var(--ink)", primary)
        self.assertIn("color: var(--paper)", primary)
        send = body_with(self.css, ".composer .icon-btn.primary", "background: var(--ink)")
        self.assertIn("color: var(--paper)", send)
        hover = body_with(self.css, "button.primary:hover", "background: #2a2624")
        self.assertIn("border-color: #2a2624", hover)
        disabled = body_with(self.css, "button.primary:disabled", "background: #b9b6ae")
        self.assertIn("opacity: 1", disabled)
        self.assertNotIn("drop-shadow", self.css)

    def test_focus_uses_ink_outline_and_active_session_uses_inversion(self) -> None:
        for selector in ("button:focus-visible", "input:focus-visible", "textarea:focus-visible", "select:focus-visible"):
            body = body_with(self.css, selector, "outline: 2px solid #141111")
            self.assertIn("outline-offset: 1px", body)
        active = body_with(self.css, ".session.active", "background: var(--ink)")
        self.assertIn("color: var(--paper)", active)
        self.assertNotIn("outline:", active)
        content = body_with(self.css, ".session.active .sessionContent", "background: var(--ink)")
        self.assertIn("color: var(--paper)", content)
        idle = body_with(self.css, ".session.active .stateDot.idle", "border-color: var(--paper)")
        self.assertIn("border-color: var(--paper)", idle)
        busy = body_with(self.css, ".session.active .stateDot.busy", "background: var(--paper)")
        self.assertIn("border-color: var(--paper)", busy)
        pending = body_with(self.css, ".session.active .stateDot.pending", "background: #f59e0b")
        self.assertIn("border-color: #f59e0b", pending)
        shadow_values = re.findall(r"box-shadow\s*:\s*([^;]+);", COMMENT_RE.sub("", self.css))
        self.assertTrue(shadow_values)
        self.assertEqual({value.strip() for value in shadow_values}, {"none"})

    def test_chrome_is_visually_32px_on_every_viewport(self) -> None:
        for selector in CHROME_SELECTORS:
            body = body_with(self.css, selector, "width: var(--ctl-chrome)")
            for declaration in (
                "height: var(--ctl-chrome)",
                "min-width: var(--ctl-chrome)",
                "min-height: var(--ctl-chrome)",
            ):
                self.assertIn(declaration, body)
        self.assertNotRegex(self.css, r"\.chatNavRail \.icon-btn\s*\{[^}]*\b(?:28|30|34)px")
        self.assertNotRegex(self.css, r"\.topActions \.icon-btn\s*\{[^}]*\b(?:34|36)px")

    def test_touch_token_and_hit_slop_produce_44px_targets(self) -> None:
        touch_root = body_with(self.touch, ":root", "--ctl: 44px")
        self.assertIn("--ctl: 44px", touch_root)
        composer = body_with(self.touch, ".composer", "--composerCtl: 44px")
        self.assertIn("--composerCtl: 44px", composer)
        button = body_with(self.touch, "button", "min-height: 44px")
        self.assertIn("touch-action: manipulation", button)
        for selector in CHROME_SELECTORS:
            self.assertIn("position: relative", body_with(self.touch, selector, "position: relative"))
            pseudo = body_with(self.touch, f"{selector}::after", 'content: ""')
            self.assertIn("position: absolute", pseudo)
            self.assertIn("inset: -6px", pseudo)

    def test_mobile_sprawl_does_not_redefine_control_shapes(self) -> None:
        for forbidden in (
            "--composerCtl",
            ".code-copy-btn",
            ".composer .icon-btn",
            ".topActions .icon-btn",
            ".chatNavRail .icon-btn",
            ".agentBackendTab",
        ):
            self.assertNotIn(forbidden, self.phone)
        self.assertNotRegex(self.phone, r"(?m)^\s*button\s*\{")

    def test_interrupt_and_chat_navigation_semantics_are_preserved(self) -> None:
        self.assertNotRegex(self.css, r"#interruptBtn\s*\{[^}]*display\s*:\s*none")
        self.assertNotIn(".composerStopBtn", self.css)
        rail = body_with(self.css, ".chatNavRail", "border: 1px solid var(--border)")
        self.assertIn("background: var(--panel)", rail)
        self.assertEqual(selector_bodies(self.touch, ".chatNavRail"), [])
        divider = body_with(self.css, ".chatMessageNavControls", "border-left: 1px solid var(--hairline)")
        self.assertIn("border-left: 1px solid var(--hairline)", divider)
        self.assertEqual(selector_bodies(self.touch, ".chatMessageNavControls"), [])

    def test_data_surfaces_use_monospace_typography(self) -> None:
        for selector in (".metaText", ".status-chip", "#ctxChip", ".chatTimeChip", ".ts"):
            self.assertIn("font-family: var(--font-mono)", body_with(self.css, selector, "font-family: var(--font-mono)"))

    def test_hint_badge_uses_solid_paper_border_treatment(self) -> None:
        source = APP_HINT_MODE.read_text(encoding="utf-8")
        self.assertIn('borderRadius: "0"', source)
        self.assertIn('border: "1px solid #141111"', source)
        self.assertIn('background: "#141111"', source)
        self.assertNotIn("boxShadow", source)

    def test_design_audit_paper_surfaces_are_bounded_and_inked(self) -> None:
        shell = APP_SHELL.read_text(encoding="utf-8")
        self.assertIn('<svg class="sidebarLogo"', shell)
        self.assertIn('fill="currentColor"', shell)
        self.assertNotIn('class="sidebarLogo" src=', shell)
        self.assertNotIn('codoxear-icon.png', shell)

        picker = body_with(self.css, ".composer .modelPicker", "width: min(92vw, 420px)")
        self.assertIn("max-width: min(92vw, 420px)", picker)
        self.assertIn("max-height: min(50vh, 320px)", picker)
        self.assertIn("overflow-y: auto", picker)
        self.assertIn("border: 1px solid var(--border)", picker)
        self.assertIn("background: var(--paper)", picker)

        details = body_with(self.css, ".detailsValue", "font-family: var(--font-mono)")
        self.assertIn("overflow-wrap: anywhere", details)
        viewer = body_with(self.css, ".diagViewer", "max-height: min(calc(var(--appH, 100dvh) - 24px), 720px)")
        self.assertIn("overflow: hidden", viewer)
        detail_scroller = body_with(self.css, ".diagViewer .detailsGrid", "overflow-y: auto")
        self.assertIn("min-height: 0", detail_scroller)
        self.assertRegex(
            self.css,
            r"@media \(max-width: 700px\) \{\s*\.detailsRow \{\s*grid-template-columns: minmax\(0, 1fr\);",
        )

        table = body_with(self.css, ".md table", "border: 1px solid var(--border)")
        self.assertIn("background: var(--paper)", table)
        header = body_with(self.css, ".md th", "background: var(--wash)")
        self.assertIn("border-bottom-color: var(--border)", header)
        cell = body_with(self.css, ".md td", "background: var(--paper)")
        self.assertIn("background: var(--paper)", cell)
        typing = body_with(self.css, ".typingDot", "background: var(--ink)")
        self.assertIn("animation: typingDot 1.2s infinite ease-in-out", typing)


if __name__ == "__main__":
    unittest.main()
