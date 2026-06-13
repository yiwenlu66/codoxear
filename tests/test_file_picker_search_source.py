import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def js_function(source: str, name: str) -> str:
    raw_start = source.index(f"function {name}")
    start = raw_start - len("async ") if source[max(0, raw_start - len("async ")) : raw_start] == "async " else raw_start
    params_end = source.index(")", raw_start)
    brace = source.index("{", params_end)
    depth = 0
    for idx in range(brace, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"could not extract {name}")


def eval_file_picker_search_helpers(state: dict) -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function normalizeFileCandidateSource(source) {")
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


def eval_resolve_file_open_mode_cases() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = js_function(source, "resolveFileOpenMode")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          fileEntryMap: new Map(),
          fileCandidateGitStateFresh: false,
          fileNonDiffMode: "file",
          inspectedKind: "text",
          inspectSessionFilePath: async () => {{ return {{ exists: true, kind: ctx.inspectedKind }}; }},
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        (async () => {{
          ctx.fileEntryMap.set("changed.py", {{ changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          ctx.inspectedKind = "text";
          const freshChanged = await ctx.resolveFileOpenMode("changed.py");
          ctx.fileCandidateGitStateFresh = false;
          const cachedChanged = await ctx.resolveFileOpenMode("changed.py");
          const explicitCachedChanged = await ctx.resolveFileOpenMode("changed.py", {{ changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          const freshExplicitUnchanged = await ctx.resolveFileOpenMode("changed.py", {{ changed: false }});
          ctx.fileCandidateGitStateFresh = false;
          ctx.fileNonDiffMode = "preview";
          ctx.inspectedKind = "markdown";
          const staleMarkdownPreview = await ctx.resolveFileOpenMode("README.md", {{ changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          ctx.inspectedKind = "image";
          const freshChangedNondiffable = await ctx.resolveFileOpenMode("image.png", {{ changed: true }});
          process.stdout.write(JSON.stringify({{
            freshChanged,
            cachedChanged,
            explicitCachedChanged,
            freshExplicitUnchanged,
            staleMarkdownPreview,
            freshChangedNondiffable,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_file_candidate_cache_helpers() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    names = [
        "listFromFilesField",
        "parseFileLocation",
        "stripPathLocationSuffix",
        "normalizeFileCandidateSource",
        "cloneFileCandidateEntry",
        "applyFileCandidateEntries",
        "currentFileCandidateEntries",
        "collectMessageFileRefs",
        "fileCandidateCacheKey",
        "rememberFileCandidateCache",
        "sessionRelativePath",
        "refreshFileCandidates",
    ]
    snippet = "\n".join(js_function(source, name) for name in names)
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          fileCandidateList: [],
          fileEntryMap: new Map(),
          fileCandidateGitStateFresh: false,
          fileCandidateCache: new Map(),
          FILE_CANDIDATE_CACHE_TTL_MS: 15000,
          fileViewerSessionId: "s1",
          selected: "",
          sessionIndex: new Map([["s1", {{ cwd: "/repo", files: ["/repo/recent.txt"] }}]]),
          chatInner: {{ querySelectorAll() {{ return []; }} }},
          renderCount: 0,
          apiCalls: 0,
        }};
        vm.createContext(ctx);
        vm.runInContext(`
          async function api(path) {{
            apiCalls += 1;
            return {{ entries: [{{ path: "changed.py", additions: 1, deletions: 0 }}] }};
          }}
          function renderFilePickerMenu() {{
            renderCount += 1;
          }}
        ` + {json.dumps(snippet)}, ctx);
        (async () => {{
          await ctx.refreshFileCandidates();
          const first = ctx.fileCandidateList.slice();
          const firstFresh = ctx.fileCandidateGitStateFresh;
          await ctx.refreshFileCandidates();
          const second = ctx.fileCandidateList.slice();
          const secondFresh = ctx.fileCandidateGitStateFresh;
          await ctx.refreshFileCandidates({{ force: true }});
          const third = ctx.fileCandidateList.slice();
          const thirdFresh = ctx.fileCandidateGitStateFresh;
          process.stdout.write(JSON.stringify({{
            apiCalls: ctx.apiCalls,
            renderCount: ctx.renderCount,
            first,
            second,
            third,
            firstFresh,
            secondFresh,
            thirdFresh,
            cacheSize: ctx.fileCandidateCache.size,
            sources: ctx.fileCandidateList.map((path) => (ctx.fileEntryMap.get(path) || {{}}).source || ""),
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
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

    def test_loaded_search_results_are_ordered_by_score_before_candidate_membership(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileCandidateList": ["zzz/src/app.js"],
                "fileEntries": [{"path": "zzz/src/app.js", "changed": False, "source": "recent"}],
                "filePickerSearchActive": True,
                "filePickerInputValue": "src/app.js",
                "fileSearchLoadedQuery": "src/app.js",
                "fileSearchResults": [{"path": "src/app.js", "score": 12000}],
            }
        )
        self.assertEqual(result["entries"][0]["path"], "src/app.js")

    def test_no_query_entries_preserve_source_metadata(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileCandidateList": ["changed.py", "mentioned.md", "recent.txt"],
                "fileEntries": [
                    {"path": "changed.py", "changed": True, "source": "changed"},
                    {"path": "mentioned.md", "changed": False, "source": "mentioned"},
                    {"path": "recent.txt", "changed": False, "source": "recent"},
                ],
                "filePickerSearchActive": False,
                "filePickerInputValue": "",
            }
        )
        self.assertEqual([entry["source"] for entry in result["entries"]], ["changed", "mentioned", "recent"])

    def test_source_shows_pending_full_project_footer(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function localFilePickerSearchEntries(query)", source)
        self.assertIn("function prependDraftFileEntry(entries, query)", source)
        self.assertIn('text: "Searching full project..."', source)
        self.assertNotIn("if (fileSearchPendingQuery === query) return null;", source)

    def test_file_candidate_cache_reuses_same_session_key(self) -> None:
        result = eval_file_candidate_cache_helpers()
        self.assertEqual(result["apiCalls"], 2)
        self.assertEqual(result["renderCount"], 3)
        self.assertEqual(result["first"], ["changed.py", "recent.txt"])
        self.assertEqual(result["second"], result["first"])
        self.assertEqual(result["third"], result["first"])
        self.assertEqual(result["cacheSize"], 1)
        self.assertEqual(result["sources"], ["changed", "recent"])
        self.assertTrue(result["firstFresh"])
        self.assertFalse(result["secondFresh"])
        self.assertTrue(result["thirdFresh"])

    def test_resolve_file_open_mode_requires_fresh_changed_metadata_for_diff(self) -> None:
        result = eval_resolve_file_open_mode_cases()
        self.assertEqual(result["freshChanged"], "diff")
        self.assertEqual(result["cachedChanged"], "file")
        self.assertEqual(result["explicitCachedChanged"], "file")
        self.assertEqual(result["freshExplicitUnchanged"], "file")
        self.assertEqual(result["staleMarkdownPreview"], "preview")
        self.assertEqual(result["freshChangedNondiffable"], "file")

    def test_file_picker_candidate_sections_and_cache_are_present(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = (APP_JS.parent / "app.css").read_text(encoding="utf-8")
        self.assertIn("function filePickerSectionLabel(source)", source)
        self.assertIn('return "Changed files";', source)
        self.assertIn('return "Mentioned in chat";', source)
        self.assertIn('return "Recently opened";', source)
        self.assertIn("const fileCandidateCache = new Map();", source)
        self.assertIn("const FILE_CANDIDATE_CACHE_TTL_MS = 15000;", source)
        self.assertIn("fileCandidateCache.set(sid, { key, ts: Date.now(), entries: currentFileCandidateEntries() });", source)
        self.assertIn("fileCandidateCache.delete(sid);", source)
        self.assertIn(".fileMenuSection", css)


if __name__ == "__main__":
    unittest.main()
