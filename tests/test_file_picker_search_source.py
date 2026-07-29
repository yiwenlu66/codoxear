import json
import subprocess
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ROOT / "codoxear" / "static" / "app_display.js"
HELPERS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
PICKER = ROOT / "codoxear" / "static" / "app_file_picker.js"


def run_picker(body: str) -> dict:
    modules = ",\n".join(json.dumps(path.read_text(encoding="utf-8")) for path in (DISPLAY, HELPERS, PICKER))
    script = f"""
const vm = require('vm'); const ctx = {{ window: {{}} }}; vm.createContext(ctx);
for (const source of [{modules}]) vm.runInContext(source, ctx);
{body}
"""
    proc = subprocess.run(["node"], input=textwrap.dedent(script), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(proc.stdout)


class TestFilePickerSearchBehavior(unittest.TestCase):
    def test_menu_search_state_tracks_query_line_focus_and_draft_suppression(self) -> None:
        result = run_picker(
            """
const picker = ctx.window.CodoxearFilePicker;
const menu = picker.createMenuState({ normalizeLineNumber: (v) => Number(v) >= 1 ? Number(v) : null });
const opened = menu.openSearchQuery('src/app.py', { line: 8, suppressDraft: true });
const initial = menu.snapshot(); const moved = menu.moveFocus(3, 1); const line = menu.selectionLine('src/app.py'); const changed = menu.handleInput('src/a');
process.stdout.write(JSON.stringify({ frozen: Object.isFrozen(picker), opened, initial, moved, line, changed }));
"""
        )
        self.assertTrue(result["frozen"])
        self.assertTrue(result["opened"])
        self.assertTrue(result["initial"]["searchActive"])
        self.assertTrue(result["initial"]["suppressDraftQuery"])
        self.assertEqual(result["line"], 8)
        self.assertEqual(result["moved"], 1)
        self.assertEqual(result["changed"]["referenceLine"], None)
        self.assertEqual(result["changed"]["focus"], -1)

    def test_local_search_scores_candidates_and_keeps_git_collision_selectable(self) -> None:
        result = run_picker(
            r'''
const picker = ctx.window.CodoxearFilePicker;
const entries = new Map([
 ['session\u0000src/foo.py', { path: 'src/foo.py', gitPath: false, apiPath: 'session-token' }],
 ['git\u0000src/foo.py', { path: 'src/foo.py', gitPath: true, apiPath: 'git-token' }],
 ['session\u0000README.md', { path: 'README.md', gitPath: false, apiPath: '' }],
]);
const context = { candidateKeys: [...entries.keys()], entryForKey: (key) => entries.get(key), pickerEntryForKey: (key, { score }) => ({ ...entries.get(key), key, score }) };
const found = picker.localFilePickerSearchEntries(context, 'foo');
const pending = picker.prependPendingSessionPathEntry(found, 'src/foo.py');
process.stdout.write(JSON.stringify({ found, pending }));
'''
        )
        self.assertEqual([entry["apiPath"] for entry in result["found"]], ["session-token", "git-token"])
        self.assertGreater(result["found"][0]["score"], 0)
        self.assertEqual(result["pending"][0]["path"], "src/foo.py")
        self.assertFalse(result["pending"][0]["gitPath"])

    def test_search_state_discards_stale_results_and_exposes_pending_query(self) -> None:
        result = run_picker(
            r'''
const picker = ctx.window.CodoxearFilePicker; const callbacks = [];
const search = picker.createSearchState({ blocked: () => false, currentSessionId: () => 'sid-1', api: async () => ({ matches: [{ path: 'needle.py' }] }), setTimeout(fn) { callbacks.push(fn); return callbacks.length; }, clearTimeout() {}, isMenuOpen: () => true, inputValue: () => 'needle', renderMenu() {}, applyMenuState() {}, normalizeFileApiPath: (value) => String(value || '') });
search.setSessionId('sid-1'); const pending = search.request('needle'); const before = search.snapshot(); pending.then(() => process.stdout.write(JSON.stringify({ before, after: search.snapshot() })));
'''
        )
        self.assertEqual(result["before"]["pendingQuery"], "needle")
        self.assertEqual(result["after"]["loadedQuery"], "needle")
        self.assertEqual(result["after"]["results"], [{"path": "needle.py", "score": 0, "apiPath": ""}])


if __name__ == "__main__":
    unittest.main()
