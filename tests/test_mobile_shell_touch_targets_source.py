import re
import unittest
from pathlib import Path


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"

SHELL_TARGET_SELECTORS = [
    ".pill > .icon-btn",
    ".topActions .icon-btn",
    ".sidebar header .icon-btn",
    ".sessionContextBar .icon-btn",
    ".chatNavRail .icon-btn",
    ".agentBackendTab",
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


def last_selector_body(css: str, selector: str) -> str:
    bodies = selector_bodies(css, selector)
    if not bodies:
        raise AssertionError(f"missing CSS rule for {selector}")
    return bodies[-1]


class TestMobileShellTouchTargetsSource(unittest.TestCase):
    def test_phone_shell_targets_have_explicit_44px_floor(self) -> None:
        """Always-used shell command surfaces must be 44x44 CSS px on phones.

        These selectors are intentionally narrower than the global .icon-btn
        class so dense desktop/editor controls keep compact sizing while the
        phone shell controls meet the touch-target floor.
        """
        css = APP_CSS.read_text(encoding="utf-8")
        phone_block = media_block(css, "@media (max-width: 520px)")

        for selector in SHELL_TARGET_SELECTORS:
            with self.subTest(selector=selector):
                body = last_selector_body(phone_block, selector)
                self.assertIn("width: 44px", body)
                self.assertIn("height: 44px", body)
                self.assertIn("min-width: 44px", body)
                self.assertIn("min-height: 44px", body)

    def test_phone_floor_overrides_existing_compact_shell_selectors(self) -> None:
        """The 44px phone floor must beat the compact 34px rules it fixes.

        sessionContextBar/chatNavRail and agentBackendTab have base 34px rules;
        topActions has a phone 34px compact rule. The phone floor appears in
        the max-width:520px block after the generic and topActions compact rules,
        and uses equal-or-higher specificity than the later coarse-pointer
        .icon-btn 40px rule.
        """
        css = APP_CSS.read_text(encoding="utf-8")
        phone_block = media_block(css, "@media (max-width: 520px)")
        coarse_block = media_block(css, "@media (hover: none) and (pointer: coarse)")

        generic_mobile = selector_bodies(phone_block, ".icon-btn")
        self.assertTrue(generic_mobile)
        self.assertIn("width: 34px", generic_mobile[0])
        self.assertIn("height: 34px", generic_mobile[0])

        top_actions_mobile = selector_bodies(phone_block, ".topActions .icon-btn")
        self.assertGreaterEqual(len(top_actions_mobile), 2)
        self.assertIn("width: 34px", top_actions_mobile[0])
        self.assertIn("height: 34px", top_actions_mobile[0])
        self.assertIn("width: 44px", top_actions_mobile[-1])
        self.assertLess(
            phone_block.index(".topActions .icon-btn"),
            phone_block.rindex(".topActions .icon-btn"),
        )

        before_phone = css[: css.index("@media (max-width: 520px)")]
        rail_body = last_selector_body(before_phone, ".sessionContextBar .icon-btn")
        self.assertIn("width: 34px", rail_body)
        self.assertIn("height: 34px", rail_body)
        self.assertIn("width: 44px", last_selector_body(phone_block, ".sessionContextBar .icon-btn"))
        self.assertIn("width: 44px", last_selector_body(phone_block, ".chatNavRail .icon-btn"))

        backend_base = last_selector_body(before_phone, ".agentBackendTab")
        self.assertIn("width: 34px", backend_base)
        self.assertIn("height: 34px", backend_base)
        self.assertIn("width: 44px", last_selector_body(phone_block, ".agentBackendTab"))

        coarse_icon = last_selector_body(coarse_block, ".icon-btn")
        self.assertIn("width: 40px", coarse_icon)
        self.assertIn("height: 40px", coarse_icon)

    def test_base_compact_sizing_remains_for_desktop_and_dense_controls(self) -> None:
        """Desktop/base compact controls must remain compact outside phones."""
        css = APP_CSS.read_text(encoding="utf-8")
        before_phone = css[: css.index("@media (max-width: 520px)")]
        phone_block = media_block(css, "@media (max-width: 520px)")

        base_icon = last_selector_body(before_phone, ".icon-btn")
        self.assertIn("width: 38px", base_icon)
        self.assertIn("height: 38px", base_icon)

        base_rail = last_selector_body(before_phone, ".chatNavRail .icon-btn")
        self.assertIn("width: 34px", base_rail)
        self.assertIn("height: 34px", base_rail)

        base_backend = last_selector_body(before_phone, ".agentBackendTab")
        self.assertIn("width: 34px", base_backend)
        self.assertIn("height: 34px", base_backend)

        # The phone floor must not be implemented by globally raising every
        # mobile .icon-btn; dense/mobile-specific controls retain their own
        # source rules unless explicitly targeted.
        mobile_generic_icon = selector_bodies(phone_block, ".icon-btn")[0]
        self.assertIn("width: 34px", mobile_generic_icon)
        self.assertIn("height: 34px", mobile_generic_icon)
        self.assertNotIn("min-width: 44px", mobile_generic_icon)
        self.assertNotIn("min-height: 44px", mobile_generic_icon)


if __name__ == "__main__":
    unittest.main()
