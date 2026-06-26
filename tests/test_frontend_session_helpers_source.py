import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"


def eval_session_helpers() -> dict:
    source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);
        const helpers = ctx.window.CodoxearSessionHelpers;
        const sessions = [
          {{ session_id: "failed", launch_state: "failed", launch_error: "boom" }},
          {{ session_id: "blocked", blocked: true }},
          {{ session_id: "snoozed", snoozed: true }},
          {{ session_id: "now", owned: true, service_tier: " fast " }},
        ];
        const entries = helpers.sidebarSessionEntries(sessions);
        process.stdout.write(JSON.stringify({{
          groups: helpers.SESSION_SIDEBAR_GROUPS,
          failedKind: helpers.sessionLaunchKind(sessions[0]),
          failedIcon: helpers.sessionLaunchIcon(sessions[0]),
          pendingSelectable: helpers.sessionSelectable({{ launch_state: "starting" }}),
          failedSelectable: helpers.sessionSelectable(sessions[0]),
          nullSelectable: helpers.sessionSelectable(null),
          tmuxKind: helpers.sessionLaunchKind({{ transport: "tmux" }}),
          webKind: helpers.sessionLaunchKind({{ owned: true }}),
          terminalKind: helpers.sessionLaunchKind({{}}),
          tmuxIcon: helpers.sessionLaunchIcon({{ transport: "tmux" }}),
          webIcon: helpers.sessionLaunchIcon({{ owned: true }}),
          terminalIcon: helpers.sessionLaunchIcon({{}}),
          reviewKey: helpers.sessionSidebarGroupKey(sessions[0]),
          waitingKey: helpers.sessionSidebarGroupKey(sessions[1]),
          laterKey: helpers.sessionSidebarGroupKey(sessions[2]),
          nowKey: helpers.sessionSidebarGroupKey(sessions[3]),
          entries,
          signature: helpers.sidebarRenderSignature(entries, {{ selectedId: "now", swipeActions: true }}),
          fast: helpers.sessionIsFast(sessions[3]),
          notFast: helpers.sessionIsFast({{ service_tier: "standard" }}),
          diagnosticsPiProvider: helpers.diagnosticsProviderDisplay({{ model_provider: "anthropic", provider_choice: "chatgpt" }}, "pi"),
          diagnosticsCcProvider: helpers.diagnosticsProviderDisplay({{ model_provider: "anthropic", provider_choice: "anthropic" }}, "cc"),
          diagnosticsCodexProvider: helpers.diagnosticsProviderDisplay({{ model_provider: "openai", provider_choice: "chatgpt" }}, "codex"),
          diagnosticsCopyText: helpers.diagnosticsCopyText("sid-1", [["CWD", "/tmp/repo"], ["Empty", ""]]),
          frozen: Object.isFrozen(helpers),
          groupsFrozen: Object.isFrozen(helpers.SESSION_SIDEBAR_GROUPS),
          groupObjectsFrozen: helpers.SESSION_SIDEBAR_GROUPS.every((group) => Object.isFrozen(group)),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendSessionHelpersSource(unittest.TestCase):
    def test_index_loads_session_helpers_before_app(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__', source)
        self.assertLess(source.index('app_file_helpers.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__'))
        self.assertLess(source.index('app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__'), source.index('app.js?v=__CODOXEAR_ASSET_VERSION__'))

    def test_app_js_requires_session_helpers_without_fallback(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        self.assertIn("const codoxearSessionHelpers = window.CodoxearSessionHelpers;", source)
        self.assertIn('throw new Error("Codoxear session helpers failed to load")', source)
        self.assertIn("const SESSION_SIDEBAR_GROUPS = codoxearSessionHelpers.SESSION_SIDEBAR_GROUPS;", source)
        for helper in [
            "sessionLaunchFailed",
            "sessionLaunchPending",
            "sessionLaunchKind",
            "sessionLaunchIcon",
            "sessionNeedsReview",
            "sessionSidebarGroupKey",
            "sidebarSessionEntries",
            "sidebarRenderSignature",
            "sessionSelectable",
            "sessionIsFast",
            "diagnosticsProviderDisplay",
            "diagnosticsCopyText",
        ]:
            self.assertIn(f"typeof codoxearSessionHelpers.{helper} !== \"function\"", source)
            self.assertIn(f"function {helper}", source)
        self.assertIn("window.CodoxearSessionHelpers = Object.freeze({", helper_source)
        self.assertIn("const SESSION_SIDEBAR_GROUPS = Object.freeze([", helper_source)
        self.assertNotIn("redactedLaunchErrorText", helper_source)
        self.assertNotIn("sessionLaunchLabel", helper_source)
        self.assertNotIn("function sidebarSessionEntries(sessions) {\n        const buckets", source)
        self.assertNotIn("function sessionIsFast(s) {\n        return !!(s && typeof s.service_tier", source)

    def test_session_helpers_preserve_grouping_and_launch_contracts(self) -> None:
        result = eval_session_helpers()
        self.assertEqual([group["key"] for group in result["groups"]], ["review", "now", "waiting", "later"])
        self.assertEqual([group["label"] for group in result["groups"]], ["Needs review", "Now", "Waiting", "Later"])
        self.assertEqual(result["failedKind"], "failed")
        self.assertEqual(result["failedIcon"], "info")
        self.assertFalse(result["pendingSelectable"])
        self.assertTrue(result["failedSelectable"])
        self.assertFalse(result["nullSelectable"])
        self.assertEqual(result["tmuxKind"], "web_tmux")
        self.assertEqual(result["webKind"], "web")
        self.assertEqual(result["terminalKind"], "terminal")
        self.assertEqual(result["tmuxIcon"], "tmux")
        self.assertEqual(result["webIcon"], "web")
        self.assertEqual(result["terminalIcon"], "terminal")
        self.assertEqual(result["reviewKey"], "review")
        self.assertEqual(result["waitingKey"], "waiting")
        self.assertEqual(result["laterKey"], "later")
        self.assertEqual(result["nowKey"], "now")
        self.assertEqual([entry["type"] for entry in result["entries"]], ["header", "session", "header", "session", "header", "session", "header", "session"])
        self.assertIn('"selectedId":"now"', result["signature"])
        self.assertIn('"swipeActions":true', result["signature"])
        self.assertTrue(result["fast"])
        self.assertFalse(result["notFast"])
        self.assertEqual(result["diagnosticsPiProvider"], "anthropic")
        self.assertEqual(result["diagnosticsCcProvider"], "-")
        self.assertEqual(result["diagnosticsCodexProvider"], "chatgpt")
        self.assertEqual(result["diagnosticsCopyText"], "Codoxear session details\nSession: sid-1\nCWD: /tmp/repo\nEmpty: -")
        self.assertTrue(result["frozen"])
        self.assertTrue(result["groupsFrozen"])
        self.assertTrue(result["groupObjectsFrozen"])


if __name__ == "__main__":
    unittest.main()
