import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_UNATTENDED_JS = ROOT / "codoxear" / "static" / "app_unattended.js"


class TestUnattendedInputSource(unittest.TestCase):
    def test_app_delegates_number_input_draft_handling_to_module(self) -> None:
        app = APP_JS.read_text(encoding="utf-8")
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        # app.js no longer owns the draft state or handlers.
        self.assertNotIn("let unattendedNumberDraft = ", app)
        self.assertNotIn("let unattendedNumberDirty = ", app)
        self.assertNotIn("function parseUnattendedDraftInt", app)
        self.assertIn("createUnattendedController", app)

    def test_unattended_number_inputs_keep_local_drafts_while_editing(self) -> None:
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        self.assertIn('let unattendedNumberDraft = { cooldown_minutes: "5", remaining_injections: "10" };', module)
        self.assertIn("let unattendedNumberDirty = { cooldown_minutes: false, remaining_injections: false };", module)
        self.assertIn("function parseUnattendedDraftInt(name) {", module)
        self.assertIn("function syncUnattendedNumberInputs() {", module)
        self.assertIn("!unattendedNumberDirty.cooldown_minutes", module)
        self.assertIn("!unattendedNumberDirty.remaining_injections", module)
        self.assertIn('finalizeUnattendedNumberDraft("cooldown_minutes");', module)
        self.assertIn('finalizeUnattendedNumberDraft("remaining_injections");', module)

    def test_invalid_empty_unattended_number_inputs_restore_last_saved_value_on_blur(self) -> None:
        module = APP_UNATTENDED_JS.read_text(encoding="utf-8")
        self.assertIn("cooldownEl.onblur = () => {", module)
        self.assertIn('restoreUnattendedNumberDraft("cooldown_minutes");', module)
        self.assertIn("remainingEl.onblur = () => {", module)
        self.assertIn('restoreUnattendedNumberDraft("remaining_injections");', module)


if __name__ == "__main__":
    unittest.main()
