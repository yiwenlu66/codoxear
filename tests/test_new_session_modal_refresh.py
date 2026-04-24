import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestNewSessionModalRefresh(unittest.TestCase):
    def test_refresh_sessions_does_not_rebuild_backend_tabs_while_modal_is_open(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('if (newSessionViewer.style.display === "flex") {')
        end = source.index("fileRefCandidateCache.clear();", start)
        block = source[start:end]
        self.assertNotIn("renderNewSessionBackendTabs();", block)
        self.assertIn("renderNewSessionProviderMenu();", block)
        self.assertIn("renderNewSessionReasoningMenu();", block)


if __name__ == "__main__":
    unittest.main()
