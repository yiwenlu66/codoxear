import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


class TestComposerSendabilitySource(unittest.TestCase):
    """A selected session keeps its normal composer when a prior delivery is
    uncertain. Recovery state is backend truth, not a frontend input mode."""

    def test_sync_composer_state_blocks_only_missing_or_failed_launch_sessions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn("function syncComposerState() {", source)
        self.assertIn('const composerInput = $("#msg");', source)
        self.assertIn("const composerBlocked = !selected || launchFailed;", source)
        self.assertIn('"Select a session to send"', source)
        self.assertIn('"Failed launch cannot receive messages"', source)
        self.assertIn('"Enter your instructions here"', source)
        self.assertIn("composerInput.disabled = composerBlocked;", source)
        self.assertIn('composerInput.setAttribute("aria-label", composerLabel);', source)
        self.assertIn('const composerPh = $("#msgPh");', source)
        self.assertIn("composerPh.textContent = composerLabel;", source)

    def test_composer_recovery_states_remain_sendable(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        composer_block = source[source.index("function syncComposerState() {") :]
        composer_block = composer_block[: composer_block.index("async function enqueueComposerText")]

        for recovery_state in ("unknownSend", "orphanRecovery", "recoveryQueue"):
            self.assertNotIn(recovery_state, composer_block)
        self.assertNotIn("commit_unknown_send", composer_block)
        self.assertNotIn("queue_recovery", composer_block)
        self.assertNotIn("orphan_recovery", composer_block)

        send_start = source.index("async function sendText(raw")
        send_block = source[send_start : source.index("const localAttachmentCount", send_start)]
        self.assertNotIn("commit_unknown_send", send_block)
        self.assertNotIn("queue_recovery", send_block)
        self.assertNotIn("orphan_recovery", send_block)
        self.assertIn('setToast(`send error: ${e2.message}`);', source)

    def test_sync_composer_state_is_driven_by_send_button_sync(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('sendControl.setAttribute("aria-label", sendLabel);\n          syncComposerState();\n        }', source)

    def test_composer_remains_editable_during_transient_send(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        composer_block = source[source.index("function syncComposerState() {") :]
        composer_block = composer_block[: composer_block.index("async function enqueueComposerText")]
        self.assertNotIn("!!sending", composer_block)

    def test_disabled_composer_has_visual_styling(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".composer textarea:disabled {", css)
        self.assertIn("opacity: 0.5;", css)
        self.assertIn("cursor: not-allowed;", css)


if __name__ == "__main__":
    unittest.main()
