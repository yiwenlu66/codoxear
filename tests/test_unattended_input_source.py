import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestUnattendedInputSource(unittest.TestCase):
    def test_unattended_number_inputs_keep_local_drafts_while_editing(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('let unattendedNumberDraft = { cooldown_minutes: "5", remaining_injections: "10" };', source)
        self.assertIn("let unattendedNumberDirty = { cooldown_minutes: false, remaining_injections: false };", source)
        self.assertIn("function parseUnattendedDraftInt(name) {", source)
        self.assertIn("function syncUnattendedNumberInputs() {", source)
        self.assertIn("!unattendedNumberDirty.cooldown_minutes", source)
        self.assertIn("!unattendedNumberDirty.remaining_injections", source)
        self.assertIn('finalizeUnattendedNumberDraft("cooldown_minutes");', source)
        self.assertIn('finalizeUnattendedNumberDraft("remaining_injections");', source)

    def test_invalid_empty_unattended_number_inputs_restore_last_saved_value_on_blur(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("unattendedCooldownEl.onblur = () => {", source)
        self.assertIn('restoreUnattendedNumberDraft("cooldown_minutes");', source)
        self.assertIn("unattendedRemainingEl.onblur = () => {", source)
        self.assertIn('restoreUnattendedNumberDraft("remaining_injections");', source)


if __name__ == "__main__":
    unittest.main()
