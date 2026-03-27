import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestHarnessInputSource(unittest.TestCase):
    def test_harness_number_inputs_keep_local_drafts_while_editing(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('let harnessNumberDraft = { cooldown_minutes: "5", remaining_injections: "10" };', source)
        self.assertIn("let harnessNumberDirty = { cooldown_minutes: false, remaining_injections: false };", source)
        self.assertIn("function parseHarnessDraftInt(name) {", source)
        self.assertIn("function syncHarnessNumberInputs() {", source)
        self.assertIn("!harnessNumberDirty.cooldown_minutes", source)
        self.assertIn("!harnessNumberDirty.remaining_injections", source)
        self.assertIn('finalizeHarnessNumberDraft("cooldown_minutes");', source)
        self.assertIn('finalizeHarnessNumberDraft("remaining_injections");', source)

    def test_invalid_empty_harness_number_inputs_restore_last_saved_value_on_blur(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("harnessCooldownEl.onblur = () => {", source)
        self.assertIn('restoreHarnessNumberDraft("cooldown_minutes");', source)
        self.assertIn("harnessRemainingEl.onblur = () => {", source)
        self.assertIn('restoreHarnessNumberDraft("remaining_injections");', source)


if __name__ == "__main__":
    unittest.main()
