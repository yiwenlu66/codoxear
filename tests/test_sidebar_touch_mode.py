import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_use_desktop_session_actions(query_matches: dict[str, bool]) -> bool:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function useDesktopSessionActions() {")
    end = source.index("function setSidebarOpen(open) {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const queryMatches = {json.dumps(query_matches)};
        const ctx = {{
          window: {{
            matchMedia: (query) => ({{ matches: Boolean(queryMatches[query]) }}),
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_useDesktopSessionActions = useDesktopSessionActions;\n")}, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_useDesktopSessionActions()));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestSidebarTouchMode(unittest.TestCase):
    def test_desktop_session_actions_require_hover_capable_wide_pointer(self) -> None:
        self.assertTrue(
            eval_use_desktop_session_actions(
                {"(hover: hover) and (pointer: fine) and (min-width: 881px)": True}
            )
        )
        self.assertFalse(
            eval_use_desktop_session_actions(
                {"(hover: hover) and (pointer: fine) and (min-width: 881px)": False}
            )
        )

    def test_refresh_sessions_uses_touch_mode_for_non_desktop_inputs(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const swipeActions = !useDesktopSessionActions();", source)
        self.assertIn("if (swipeActions && openSwipeSessionId && sessionsWrap.childElementCount > 0) {", source)
        self.assertIn("if (swipeActions) {", source)
        self.assertIn("On touch devices: swipe left on a session", source)


if __name__ == "__main__":
    unittest.main()
