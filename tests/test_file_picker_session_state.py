import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_file_picker_session_helpers() -> dict[str, object]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function currentFileSessionId() {")
    end = source.index("function extToEditorLang(p) {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          selected: "session-a",
          fileViewerSessionId: "session-a",
          identity: {{ path: "file-a.py", apiPath: "token-a", gitPath: true, line: 7 }},
          fileViewerController: {{
            currentActiveFileIdentity: () => ({{ path: ctx.identity.path, apiPath: ctx.identity.apiPath, gitPath: ctx.identity.gitPath }}),
            currentActiveFileLine: () => ctx.identity.line,
            nextActiveFileIdentity: (current, nextPath, opts = {{}}) => ({{ path: String(nextPath ?? ""), gitPath: Boolean(opts.gitPath ?? current.gitPath), apiPath: String(opts.apiPath ?? current.apiPath ?? "") }}),
            clearActiveFileIdentity: () => {{ ctx.identity = {{ path: "", apiPath: "", gitPath: false, line: null }}; }},
            beginActiveFileIdentity: (nextPath = null) => ({{ path: String(nextPath ?? ctx.identity.path), apiPath: ctx.identity.apiPath, gitPath: ctx.identity.gitPath, line: ctx.identity.line }}),
          }},
          fileSessionSelections: new Map(),
          sessionIndex: new Map([
            ["session-a", {{ cwd: "/project-A", files: ["/project-A/file-a.py"] }}],
            ["session-b", {{ cwd: "/project-B", files: ["/project-B/file-b.py"] }}],
          ]),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          normalizeLineNumber: (value) => {{
            if (value == null || value === "") return null;
            const n = Number(value);
            return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
          }},
          listFromFilesField: (value) => Array.isArray(value) ? value.slice() : [],
          sessionRelativePath: (rawPath, sidOverride = null) => {{
            const sid = sidOverride || ctx.selected;
            const session = ctx.sessionIndex.get(sid);
            if (!session || !session.cwd) return null;
            const abs = String(rawPath || "").trim();
            const cwd = String(session.cwd || "").replace(/\\/+$/, "");
            if (!abs) return null;
            if (abs === cwd) return ".";
            if (abs.startsWith(cwd + "/")) return abs.slice(cwd.length + 1);
            return null;
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_file_picker = { rememberActiveFileSelection, preferredFileSelectionForSession, historyFileSelectionForSession };\n")}, ctx);
        ctx.__test_file_picker.rememberActiveFileSelection();
        process.stdout.write(JSON.stringify({{
          sessionA: ctx.__test_file_picker.preferredFileSelectionForSession("session-a"),
          sessionB: ctx.__test_file_picker.preferredFileSelectionForSession("session-b"),
        }}));
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


class TestFilePickerSessionState(unittest.TestCase):
    def test_preferred_file_selection_is_session_scoped(self) -> None:
        result = eval_file_picker_session_helpers()
        self.assertEqual(result["sessionA"], {"path": "file-a.py", "apiPath": "token-a", "line": 7, "gitPath": True})
        self.assertEqual(result["sessionB"], {"path": "file-b.py", "line": None, "gitPath": False})

    def test_global_file_path_local_storage_is_not_used(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function ensureCurrentFileViewerSession()", source)
        self.assertNotIn('localStorage.getItem("codexweb.filePath")', source)
        self.assertNotIn('localStorage.setItem("codexweb.filePath"', source)


if __name__ == "__main__":
    unittest.main()
