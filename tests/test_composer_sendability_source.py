import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


class TestComposerSendabilitySource(unittest.TestCase):
    """The composer textarea must not advertise a normal send affordance when the
    selected session cannot receive messages. The blocked condition mirrors the
    structural send-button block (no transient `sending`/busy state), because
    enqueueComposerText is blocked by the same predicates as sendText."""

    def test_sync_composer_state_exists_and_projects_sendability(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn("function syncComposerState() {", source)
        self.assertIn('const composerInput = $("#msg");', source)
        self.assertIn("const composerBlocked = !selected || launchFailed || unknownSend || orphanRecovery || recoveryQueue;", source)
        # Reason labels must match the send-button vocabulary.
        self.assertIn('"Select a session to send"', source)
        self.assertIn('"Failed launch cannot receive messages"', source)
        self.assertIn('"Resolve the unknown send before sending"', source)
        self.assertIn('"Missing session can only be reviewed"', source)
        self.assertIn('"Review preserved queued recovery items before sending"', source)
        # Live-session default affordance is preserved.
        self.assertIn('"Enter your instructions here"', source)
        # The textarea itself is disabled (not just relabeled) when blocked.
        self.assertIn("composerInput.disabled = composerBlocked;", source)
        self.assertIn('composerInput.setAttribute("aria-label", composerLabel);', source)
        # The overlay placeholder div carries the reason text.
        self.assertIn('const composerPh = $("#msgPh");', source)
        self.assertIn("composerPh.textContent = composerLabel;", source)

    def test_sync_composer_state_is_driven_by_send_button_sync(self) -> None:
        """syncSendButtonState owns the sendability transition points (session
        select, send start/end, recovery changes), so the composer must be
        refreshed from there to stay consistent without enumerating every site."""
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('sendControl.setAttribute("aria-label", sendLabel);\n          syncComposerState();\n        }', source)

    def test_composer_blocked_condition_excludes_transient_sending(self) -> None:
        """The composer must remain editable during a live/busy turn so the
        queue/send-choice path is unaffected. `sending` must not appear in the
        composer blocked expression."""
        source = APP_JS.read_text(encoding="utf-8")
        composer_block = source[source.index("function syncComposerState() {"):]
        composer_block = composer_block[: composer_block.index("async function enqueueComposerText")]
        self.assertIn("composerBlocked = !selected || launchFailed || unknownSend || orphanRecovery || recoveryQueue;", composer_block)
        # The send button's disabled line still carries `sending`; the composer's must not.
        self.assertNotIn("!!sending", composer_block)

    def test_disabled_composer_has_visual_styling(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".composer textarea:disabled {", css)
        self.assertIn("opacity: 0.5;", css)
        self.assertIn("cursor: not-allowed;", css)


if __name__ == "__main__":
    unittest.main()
