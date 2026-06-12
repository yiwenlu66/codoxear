import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


class TestLoginAccessibilitySource(unittest.TestCase):
    def test_login_uses_form_submit_and_password_semantics(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function renderLogin(onAuthed)")
        end = source.index("function renderApp()", start)
        block = source[start:end]

        self.assertIn('const pwInput = el("input", {', block)
        self.assertIn('type: "password"', block)
        self.assertIn('name: "password"', block)
        self.assertIn('"aria-label": "Password"', block)
        self.assertIn('autocomplete: "current-password"', block)
        self.assertIn('"aria-describedby": "loginError"', block)
        self.assertIn('const loginBtn = el("button", { class: "primary", id: "loginBtn", type: "submit", text: "Login" });', block)
        self.assertIn('const form = el("form", { class: "login", id: "loginForm" }', block)
        self.assertIn('el("label", { class: "sr-only", for: "pw", text: "Password" })', block)
        self.assertIn("form.onsubmit = async (e) => {", block)
        self.assertIn("e.preventDefault();", block)
        self.assertIn("const pw = pwInput.value;", block)
        self.assertIn("pwInput.focus();", block)
        self.assertNotIn('$("#loginBtn").onclick', block)

    def test_sr_only_utility_exists_for_login_label(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".sr-only {", css)
        self.assertIn("position: absolute;", css)
        self.assertIn("clip: rect(0, 0, 0, 0);", css)


if __name__ == "__main__":
    unittest.main()
