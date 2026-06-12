import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


class TestSidebarGtdSource(unittest.TestCase):
    def test_sidebar_groups_use_existing_session_fields(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('const SESSION_SIDEBAR_GROUPS = [', source)
        self.assertIn('{ key: "review", label: "Needs review" }', source)
        self.assertIn('{ key: "now", label: "Now" }', source)
        self.assertIn('{ key: "waiting", label: "Waiting" }', source)
        self.assertIn('{ key: "later", label: "Later" }', source)
        self.assertIn('return !!(s && (sessionLaunchFailed(s) || s.orphan_recovery || s.queue_recovery || s.commit_unknown_send));', source)
        self.assertIn('if (s && s.blocked) return "waiting";', source)
        self.assertIn('if (s && s.snoozed) return "later";', source)

    def test_sidebar_grouping_is_render_only_not_collapsible(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('const sidebarEntries = sidebarSessionEntries(sessions);', source)
        self.assertIn('sessionsWrap.appendChild(renderSessionGroupHeader(entry));', source)
        self.assertIn('for (const entry of sidebarEntries)', source)
        self.assertIn('const s = entry.session;', source)
        self.assertNotIn('sessionGroupToggle', source)
        self.assertNotIn('sessionGroupCollapsed', source)

    def test_sidebar_group_css_is_sparse_header_not_card(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('.sessionGroupHeader {', css)
        self.assertIn('text-transform: uppercase;', css)
        self.assertIn('.sessionGroupCount {', css)
        self.assertNotIn('.sessionGroupHeader button', css)


if __name__ == "__main__":
    unittest.main()
