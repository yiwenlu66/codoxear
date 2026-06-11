import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


class TestChatNavigationSource(unittest.TestCase):
    def test_topbar_has_loaded_user_message_jump_buttons(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "prevUserBtn"', source)
        self.assertIn('title: "Previous user message"', source)
        self.assertIn('id: "nextUserBtn"', source)
        self.assertIn('title: "Next user message"', source)
        self.assertIn("prevUserBtn,", source)
        self.assertIn("nextUserBtn,", source)

    def test_jump_logic_is_loaded_user_rows_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('function loadedUserMessageRows() {', source)
        self.assertIn('row.dataset.role === "user"', source)
        self.assertIn('function jumpToLoadedUserMessage(direction)', source)
        self.assertIn('setToast("No loaded user messages")', source)
        self.assertIn('setToast("At first loaded user message")', source)
        self.assertIn('setToast("At last loaded user message")', source)
        self.assertIn('target.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" })', source)

    def test_jump_target_has_temporary_pulse_style(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('row.classList.add("nav-pulse")', source)
        self.assertIn(".msg-row.nav-pulse .msg", css)
        self.assertIn("@keyframes navPulse", css)


if __name__ == "__main__":
    unittest.main()
