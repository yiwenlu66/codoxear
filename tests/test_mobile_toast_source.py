import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


class TestMobileToastSource(unittest.TestCase):
    def test_toast_is_visible_feedback_on_mobile(self) -> None:
        js = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('id: "toast", role: "status", "aria-live": "polite"', js)
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        toast_start = mobile_block.index(".toast {")
        toast_end = mobile_block.index("}", toast_start)
        toast_block = mobile_block[toast_start:toast_end]
        self.assertNotIn("display: none", toast_block)
        self.assertIn("position: fixed;", toast_block)
        self.assertIn("bottom: calc(76px + env(safe-area-inset-bottom));", toast_block)
        self.assertIn("pointer-events: none;", toast_block)
        self.assertIn(".toast:empty", css)

    def test_file_viewer_toolbar_meets_44px_touch_target_on_mobile(self) -> None:
        """D3: file-viewer toolbar icon buttons must be at least 44x44 CSS px on mobile.

        The header actions (diff/preview/edit/video/download/close) are .icon-btn
        elements inside .fileViewer. The mobile block must raise them to the WCAG
        touch-target minimum without depending on the coarse-pointer override
        (which only reaches 40px) and without enlarging the .fileTouchBtn dpad
        that is laid out on a 34px grid.
        """
        css = APP_CSS.read_text(encoding="utf-8")
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        # The next top-level media query ends the max-width:520px block scope.
        next_media = mobile_block.find("@media", 1)
        mobile_block = mobile_block if next_media == -1 else mobile_block[:next_media]
        self.assertIn(".fileViewer .icon-btn", mobile_block)
        # Touch controls share the .fileTouchBtn class and sit on a fixed grid;
        # they must be excluded so the rule does not blow up the dpad layout.
        self.assertIn(".fileViewer .icon-btn:not(.fileTouchBtn)", mobile_block)
        rule_start = mobile_block.index(".fileViewer .icon-btn:not(.fileTouchBtn)")
        rule_end = mobile_block.index("}", rule_start)
        rule_block = mobile_block[rule_start:rule_end]
        self.assertIn("min-width: 44px", rule_block)
        self.assertIn("min-height: 44px", rule_block)
        # #fileEditBtn has an earlier ID-specific min-width: 38px rule; mobile
        # CSS must override it explicitly or the disabled Edit control remains
        # a sub-44px touch target despite the broader toolbar rule.
        edit_rule_start = mobile_block.index(".fileViewer #fileEditBtn")
        edit_rule_end = mobile_block.index("}", edit_rule_start)
        edit_rule_block = mobile_block[edit_rule_start:edit_rule_end]
        self.assertIn("min-width: 44px", edit_rule_block)


if __name__ == "__main__":
    unittest.main()
