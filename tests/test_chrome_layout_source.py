import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "codoxear" / "static" / "app.css").read_text(encoding="utf-8")
SHELL = (ROOT / "codoxear" / "static" / "app_shell.js").read_text(encoding="utf-8")
ROWS = (ROOT / "codoxear" / "static" / "app_message_rows.js").read_text(encoding="utf-8")
APP = (ROOT / "codoxear" / "static" / "app.js").read_text(encoding="utf-8")


RULE_RE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", re.MULTILINE)


def media_block(css: str, marker: str) -> str:
    start = css.index(marker)
    brace = css.index("{", start)
    depth = 1
    index = brace + 1
    while depth:
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
        index += 1
    return css[brace + 1 : index - 1]


def rule_body(css: str, selector: str, contains: str | None = None) -> str:
    for match in RULE_RE.finditer(css):
        if selector not in {part.strip() for part in match.group("selectors").split(",")}:
            continue
        body = match.group("body")
        if contains is None or contains in body:
            return body
    suffix = f" containing {contains!r}" if contains else ""
    raise AssertionError(f"No rule for {selector}{suffix}")


class TestChromeLayoutSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.touch = media_block(CSS, "@media (max-width: 700px), (pointer: coarse)")

    def test_message_copy_control_is_in_the_bubble_meta_row(self) -> None:
        self.assertIn('const meta = el("div", { class: "msg-meta" });', ROWS)
        self.assertLess(ROWS.index("bubble.appendChild(meta)"), ROWS.index("shell.appendChild(bubble)"))
        self.assertNotIn("shell.appendChild(copyBtn)", ROWS)
        meta = rule_body(CSS, ".msg-meta")
        self.assertIn("display: flex", meta)
        self.assertIn("justify-content: space-between", meta)
        copy = rule_body(CSS, ".msg-copy-btn")
        self.assertIn("width: 30px", copy)
        self.assertIn("height: 30px", copy)
        self.assertNotIn("position: absolute", copy)
        self.assertNotIn("right:", copy)

    def test_touch_secondary_controls_keep_visual_size_and_gain_hit_slop(self) -> None:
        copy = rule_body(self.touch, ".msg-copy-btn")
        self.assertIn("width: 30px", copy)
        self.assertIn("height: 30px", copy)
        self.assertNotIn("44px", copy)
        copy_slop = rule_body(self.touch, ".msg-copy-btn::after")
        self.assertIn("position: absolute", copy_slop)
        self.assertIn("inset: -7px", copy_slop)
        jump = rule_body(self.touch, ".jumpBtn")
        self.assertIn("width: 32px", jump)
        self.assertIn("height: 32px", jump)
        self.assertNotIn("44px", jump)
        remove = rule_body(self.touch, ".stagedAttachmentRemove")
        self.assertIn("min-width: 18px", remove)
        self.assertIn("min-height: 18px", remove)
        remove_slop = rule_body(self.touch, ".stagedAttachmentRemove::after")
        self.assertIn("inset: -7px", remove_slop)

    def test_staged_tray_is_a_full_width_composer_row(self) -> None:
        self.assertIn('el("div", { class: "composerInputRow" }', SHELL)
        self.assertIn("stagedTray,\n      el(\"div\", { class: \"composerInputRow\" }", SHELL)
        tray = rule_body(CSS, ".stagedAttachments")
        self.assertIn("width: 100%", tray)
        self.assertIn("flex-wrap: wrap", tray)
        chip = rule_body(CSS, ".stagedAttachmentChip")
        self.assertIn("height: 32px", chip)
        clear = rule_body(CSS, ".stagedAttachmentsClear", "height: 32px")
        self.assertIn("height: 32px", clear)
        self.assertIn("margin-left: auto", clear)
        self.assertIn("middleEllipsis(name)", APP)
        form = rule_body(CSS, ".composer form")
        self.assertNotIn("border:", form)
        self.assertNotIn("background:", form)
        drop = rule_body(CSS, ".composer.drop-active form")
        self.assertIn("background: var(--wash)", drop)
        self.assertNotIn("outline:", drop)

    def test_chat_header_keeps_time_and_navigation_in_flow(self) -> None:
        self.assertIn('const chatHeader = el("div", { class: "chatHeader", id: "chatHeader" }, [chatTimeChip, chatNavRail]);', SHELL)
        header = rule_body(CSS, ".chatHeader")
        self.assertIn("display: grid", header)
        self.assertIn("padding: 6px 12px 8px", header)
        time = rule_body(CSS, ".chatTimeChip", "grid-column: 2")
        self.assertIn("grid-column: 2", time)
        self.assertNotIn("position:", time)
        self.assertNotIn("border:", time)
        self.assertNotIn("background:", time)
        rail = rule_body(CSS, ".chatNavRail")
        self.assertIn("grid-column: 3", rail)
        self.assertNotIn("align-self", rail)
        self.assertNotIn("display: none !important", self.touch)


if __name__ == "__main__":
    unittest.main()
