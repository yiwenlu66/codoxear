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
        (which only reaches 40px). The .fileTouchBtn dpad is covered by its own
        44px rule (see test_file_touch_dpad_meets_44px_touch_target_on_mobile).
        """
        css = APP_CSS.read_text(encoding="utf-8")
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        # The next top-level media query ends the max-width:520px block scope.
        next_media = mobile_block.find("@media", 1)
        mobile_block = mobile_block if next_media == -1 else mobile_block[:next_media]
        self.assertIn(".fileViewer .icon-btn", mobile_block)
        # The header actions rule is scoped to exclude .fileTouchBtn because the
        # dpad has its own dedicated 44px rule (validated separately below).
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

    def test_file_touch_dpad_meets_44px_touch_target_on_mobile(self) -> None:
        """D5: the file-viewer touch dpad (.fileTouchBtn) controls must be at
        least 44x44 CSS px on mobile.

        The dpad (up/left/down/right) and the actions row (select/copy/paste)
        share the .fileTouchBtn class and were previously laid out on a 34px
        grid. The mobile block must now raise every visible .fileTouchBtn to
        the 44x44 WCAG minimum. The dpad grid tracks and spacers must also be
        44px so the buttons fill their cells without overflow. The total dpad
        width (3x44 + 2x6 gap = 144px) plus the actions row width (3x44 +
        2x6 = 144px) fit within the 370px mobile toolbar channel (390px
        viewport - 2x10px inset), so no horizontal overflow is introduced.
        """
        css = APP_CSS.read_text(encoding="utf-8")
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        next_media = mobile_block.find("@media", 1)
        mobile_block = mobile_block if next_media == -1 else mobile_block[:next_media]
        # Dpad grid tracks must be 44px (not the base 34px) so the buttons fill
        # their cells.
        dpad_start = mobile_block.index(".fileViewer .fileTouchDpad")
        dpad_end = mobile_block.index("}", dpad_start)
        dpad_block = mobile_block[dpad_start:dpad_end]
        self.assertIn("grid-template-columns: repeat(3, 44px)", dpad_block)
        self.assertIn("grid-template-rows: repeat(2, 44px)", dpad_block)
        # Spacers must match so the grid is not misaligned.
        spacer_start = mobile_block.index(".fileViewer .fileTouchSpacer")
        spacer_end = mobile_block.index("}", spacer_start)
        spacer_block = mobile_block[spacer_start:spacer_end]
        self.assertIn("width: 44px", spacer_block)
        self.assertIn("height: 44px", spacer_block)
        # Every .fileTouchBtn must be at least 44x44.
        btn_start = mobile_block.index(".fileViewer .fileTouchBtn")
        btn_end = mobile_block.index("}", btn_start)
        btn_block = mobile_block[btn_start:btn_end]
        self.assertIn("width: 44px", btn_block)
        self.assertIn("height: 44px", btn_block)
        self.assertIn("min-width: 44px", btn_block)
        self.assertIn("min-height: 44px", btn_block)
        # No-overflow intent: the dpad is a 3x2 grid of 44px tracks with 6px
        # gaps. 3*44 + 2*6 = 144px for the dpad and 144px for the actions row,
        # totaling 288px — well within the 370px mobile toolbar channel. The
        # rule must remain scoped under .fileViewer so the base .fileTouchBtn
        # 34px rule (outside any media query) is not altered for desktop.
        self.assertIn(".fileViewer .fileTouchBtn", mobile_block)
        # The base (non-mobile) .fileTouchBtn rule must still exist at 34px so
        # desktop layout is unchanged.
        base_idx = css.index("      .fileTouchBtn {")
        base_end = css.index("}", base_idx)
        base_block = css[base_idx:base_end]
        self.assertIn("width: 34px", base_block)
        self.assertIn("height: 34px", base_block)

    def test_composer_controls_meet_44px_touch_target_on_mobile(self) -> None:
        """Main composer icon controls (paperclip/queue/stop/send) must be at
        least 44x44 CSS px on mobile.

        These are the primary send/attach controls and the most-used touch
        targets in the app. They are .icon-btn elements inside .composer, sized
        by --composerCtl which the later coarse-pointer media query overrides to
        40px on phones. The mobile (max-width: 520px) block must therefore set
        an explicit min-width/min-height: 44px floor on .composer .icon-btn so
        the 44px target does not depend on the variable cascade. This also
        ensures the 44px touch rule is not unique to the file viewer.
        """
        css = APP_CSS.read_text(encoding="utf-8")
        mobile_start = css.index("@media (max-width: 520px)")
        mobile_block = css[mobile_start:]
        next_media = mobile_block.find("@media", 1)
        mobile_block = mobile_block if next_media == -1 else mobile_block[:next_media]
        self.assertIn(".composer .icon-btn", mobile_block)
        rule_start = mobile_block.index(".composer .icon-btn")
        rule_end = mobile_block.index("}", rule_start)
        rule_block = mobile_block[rule_start:rule_end]
        self.assertIn("min-width: 44px", rule_block)
        self.assertIn("min-height: 44px", rule_block)
        # The 44px touch-target floor must not be unique to the file viewer;
        # the composer (primary send/attach path) needs its own floor.
        self.assertIn(".fileViewer .icon-btn:not(.fileTouchBtn)", mobile_block)
        composer_idx = mobile_block.index(".composer .icon-btn")
        viewer_idx = mobile_block.index(".fileViewer .icon-btn:not(.fileTouchBtn)")
        self.assertNotEqual(composer_idx, viewer_idx)


if __name__ == "__main__":
    unittest.main()
