import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_file_picker_search_helpers(state: dict) -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function fileSearchScore(candidate, query) {")
    end = source.index("async function getKnownFileRefCandidates() {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const state = {json.dumps(state)};
        const ctx = {{
          fileCandidateList: state.fileCandidateList || [],
          fileEntryMap: new Map((state.fileEntries || []).map((entry) => [entry.path, entry])),
          activeFileDraft: Boolean(state.activeFileDraft),
          activeFilePath: state.activeFilePath || "",
          filePickerSearchActive: Boolean(state.filePickerSearchActive),
          filePickerInput: {{ value: state.filePickerInputValue || "" }},
          fileSearchResults: state.fileSearchResults || [],
          fileSearchLoadedQuery: state.fileSearchLoadedQuery || "",
          fileSearchPendingQuery: state.fileSearchPendingQuery || "",
          fileSearchErrorQuery: state.fileSearchErrorQuery || "",
          baseName: (path) => String(path || "").split("/").pop() || "",
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_file_picker_search = { visibleFilePickerEntries, localFilePickerSearchEntries };\n")}, ctx);
        const entries = ctx.__test_file_picker_search.visibleFilePickerEntries();
        process.stdout.write(JSON.stringify({{
          entries,
          local: ctx.__test_file_picker_search.localFilePickerSearchEntries(state.filePickerInputValue || ""),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFilePickerSearchSource(unittest.TestCase):
    def test_pending_search_returns_local_fuzzy_results(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileCandidateList": ["src/server.py", "codoxear/static/app.js", "README.md"],
                "fileEntries": [
                    {"path": "src/server.py", "changed": True, "additions": 5, "deletions": 1},
                    {"path": "codoxear/static/app.js", "changed": False},
                    {"path": "README.md", "changed": False},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "server",
                "fileSearchPendingQuery": "server",
                "fileSearchLoadedQuery": "",
            }
        )
        entries = result["entries"]
        self.assertIsInstance(entries, list)
        self.assertTrue(any(entry["path"] == "src/server.py" for entry in entries))
        self.assertNotEqual(entries, [])
        self.assertGreaterEqual(next(entry["score"] for entry in entries if entry["path"] == "src/server.py"), 0)

    def test_error_search_keeps_local_results_available(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileCandidateList": ["src/server.py"],
                "fileEntries": [{"path": "src/server.py", "changed": False}],
                "filePickerSearchActive": True,
                "filePickerInputValue": "server",
                "fileSearchErrorQuery": "server",
                "fileSearchError": "network down",
            }
        )
        self.assertTrue(any(entry["path"] == "src/server.py" for entry in result["entries"]))

    def test_source_shows_pending_full_project_footer(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function localFilePickerSearchEntries(query)", source)
        self.assertIn("function prependDraftFileEntry(entries, query)", source)
        self.assertIn('text: "Searching full project..."', source)
        self.assertNotIn("if (fileSearchPendingQuery === query) return null;", source)


if __name__ == "__main__":
    unittest.main()
