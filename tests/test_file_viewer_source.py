import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestFileViewerSource(unittest.TestCase):
    def test_file_editor_disables_monaco_suggestions(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("quickSuggestions: false", source)
        self.assertIn("suggestOnTriggerCharacters: false", source)
        self.assertIn('acceptSuggestionOnEnter: "off"', source)
        self.assertIn('tabCompletion: "off"', source)
        self.assertIn('wordBasedSuggestions: "off"', source)

    def test_client_reload_checks_server_ui_version(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.CODOXEAR_ASSET_VERSION", source)
        self.assertIn("function maybeReloadForUpdatedUi(nextVersion)", source)
        self.assertIn("maybeReloadForUpdatedUi(data && data.app_version)", source)
        self.assertIn("window.location.reload();", source)


if __name__ == "__main__":
    unittest.main()
