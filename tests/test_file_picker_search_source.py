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
    start = source.index("function fileCandidateKey(path")
    end = source.index("async function getKnownFileRefCandidates() {", start)
    snippet = source[start:end]
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const state = {json.dumps(state)};
        const ctx = {{
          fileCandidateList: [],
          fileEntryMap: new Map(),
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
        vm.runInContext({json.dumps(snippet + "\nglobalThis.__test_file_picker_search = { applyFileCandidateEntries, visibleFilePickerEntries, localFilePickerSearchEntries };\n")}, ctx);
        const seedEntries = state.fileEntries || (state.fileCandidateList || []).map((path) => ({{ path }}));
        ctx.__test_file_picker_search.applyFileCandidateEntries(seedEntries);
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
    snippet = "\n".join(js_function(source, name) for name in ["fileCandidateKey", "isGitFileCandidatePath", "resolveFileOpenMode"])
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          fileEntryMap: new Map(),
          fileCandidateGitStateFresh: false,
          fileNonDiffMode: "file",
          inspectedKind: "text",
          inspectCalls: [],
          inspectSessionFilePath: async (path, options = {{}}) => {{ ctx.inspectCalls.push([path, options]); return ctx.inspectedKind === "missing" ? {{ exists: false }} : {{ exists: true, kind: ctx.inspectedKind }}; }},
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        (async () => {{
          ctx.fileEntryMap.set(ctx.fileCandidateKey("changed.py", true), {{ path: "changed.py", gitPath: true, changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          ctx.inspectedKind = "text";
          const freshChanged = await ctx.resolveFileOpenMode("changed.py");
          ctx.fileCandidateGitStateFresh = false;
          const cachedChanged = await ctx.resolveFileOpenMode("changed.py");
          const explicitCachedChanged = await ctx.resolveFileOpenMode("changed.py", {{ changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          const freshExplicitUnchanged = await ctx.resolveFileOpenMode("changed.py", {{ changed: false }});
          const freshExplicitGitFalse = await ctx.resolveFileOpenMode("changed.py", {{ gitPath: false }});
          ctx.fileCandidateGitStateFresh = false;
          ctx.fileNonDiffMode = "preview";
          ctx.inspectedKind = "markdown";
          const staleMarkdownPreview = await ctx.resolveFileOpenMode("README.md", {{ changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          ctx.inspectedKind = "image";
          const freshChangedNondiffable = await ctx.resolveFileOpenMode("image.png", {{ changed: true }});
          ctx.fileCandidateGitStateFresh = true;
          ctx.inspectedKind = "text";
          const staleRememberedGitPath = await ctx.resolveFileOpenMode("stale.py", {{ gitPath: true }});
          ctx.inspectedKind = "missing";
          const deletedChanged = await ctx.resolveFileOpenMode("gone.md", {{ changed: true }});
          process.stdout.write(JSON.stringify({{
            freshChanged,
            cachedChanged,
            explicitCachedChanged,
            freshExplicitUnchanged,
            freshExplicitGitFalse,
            staleMarkdownPreview,
            freshChangedNondiffable,
            staleRememberedGitPath,
            deletedChanged,
            gitFalseInspect: ctx.inspectCalls.find((call) => call[0] === "changed.py" && call[1] && call[1].gitPath === false) || null,
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
        "fileCandidateKey",
        "fileCandidateKeyForEntry",
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
          fileCandidateRequestSeq: 0,
          FILE_CANDIDATE_CACHE_TTL_MS: 15000,
          fileViewerSessionId: "s1",
          selected: "",
          sessionIndex: new Map([["s1", {{ cwd: "/repo", files: ["/repo/recent.txt"] }}]]),
          chatInner: {{ querySelectorAll() {{ return []; }} }},
          renderCount: 0,
          applyModeCount: 0,
          apiCalls: 0,
        }};
        vm.createContext(ctx);
        vm.runInContext(`
          async function api(path) {{
            apiCalls += 1;
            return {{ entries: [{{ path: "changed.py", additions: 1, deletions: 0 }}] }};
          }}
          function applyFileMode() {{
            applyModeCount += 1;
          }}
          function renderFilePickerMenu() {{
            renderCount += 1;
          }}
          function isFileViewerSessionCurrent() {{
            return true;
          }}
          function blockUnavailableFileAction() {{
            return false;
          }}
        ` + {json.dumps(snippet)}, ctx);
        (async () => {{
          const literalFiles = ctx.listFromFilesField(["/repo/trail.md ", "/repo/new\\n.md", "", "/repo/trail.md "]);
          const literalRel = ctx.sessionRelativePath("/repo/trail.md ", "s1");
          const literalSuffix = ctx.stripPathLocationSuffix("/repo/trail.md ");
          ctx.applyFileCandidateEntries([{{ path: "trail.md ", changed: true, source: "changed" }}, {{ path: "new\\n.md", source: "recent" }}]);
          const literalCandidates = ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }}));
          ctx.sessionIndex.get("s1").files = literalFiles;
          const literalCacheKey = ctx.fileCandidateCacheKey("s1");
          ctx.sessionIndex.get("s1").files = ["/repo/recent.txt"];
          ctx.fileCandidateList = [];
          ctx.fileEntryMap = new Map();
          await ctx.refreshFileCandidates();
          const first = ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }}));
          const firstFresh = ctx.fileCandidateGitStateFresh;
          await ctx.refreshFileCandidates();
          const second = ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }}));
          const secondFresh = ctx.fileCandidateGitStateFresh;
          await ctx.refreshFileCandidates({{ force: true }});
          const third = ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }}));
          const thirdFresh = ctx.fileCandidateGitStateFresh;
          process.stdout.write(JSON.stringify({{
            apiCalls: ctx.apiCalls,
            renderCount: ctx.renderCount,
            applyModeCount: ctx.applyModeCount,
            first,
            second,
            third,
            firstFresh,
            secondFresh,
            thirdFresh,
            cacheSize: ctx.fileCandidateCache.size,
            sources: ctx.currentFileCandidateEntries().map((entry) => entry.source || ""),
            literalFiles,
            literalRel,
            literalSuffix,
            literalCandidates,
            literalCacheKey,
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
        self.assertEqual([entry["gitPath"] for entry in result["entries"]], [True, False, False])

    def test_same_display_path_keeps_git_and_session_candidates_distinct(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                    {"path": "foo.py", "changed": False, "source": "recent", "gitPath": False},
                ],
                "filePickerSearchActive": False,
                "filePickerInputValue": "",
            }
        )
        self.assertEqual(
            [(entry["path"], entry["source"], entry["gitPath"], entry["changed"]) for entry in result["entries"]],
            [("foo.py", "changed", True, True), ("foo.py", "recent", False, False)],
        )

    def test_exact_search_prefers_session_identity_over_git_collision(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "foo.py",
                "fileSearchLoadedQuery": "foo.py",
                "fileSearchResults": [{"path": "foo.py", "score": 12000}],
            }
        )
        self.assertEqual(
            [(entry["path"], entry["gitPath"], entry["changed"], entry.get("createNew", False)) for entry in result["entries"][:2]],
            [("foo.py", False, False, False), ("foo.py", True, True, False)],
        )

    def test_pending_exact_search_prefers_session_probe_over_git_path(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "foo.py",
                "fileSearchPendingQuery": "foo.py",
            }
        )
        self.assertFalse(result["entries"][0].get("createNew", False))
        self.assertTrue(result["entries"][0].get("pendingSessionPath", False))
        self.assertEqual((result["entries"][0]["path"], result["entries"][0]["gitPath"]), ("foo.py", False))
        self.assertEqual((result["entries"][1]["path"], result["entries"][1]["gitPath"]), ("foo.py", True))

    def test_pending_normalized_exact_search_prefers_session_probe_over_git_path(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "./foo.py",
                "fileSearchPendingQuery": "./foo.py",
            }
        )
        self.assertTrue(result["entries"][0].get("pendingSessionPath", False))
        self.assertEqual((result["entries"][0]["path"], result["entries"][0]["gitPath"]), ("foo.py", False))
        self.assertEqual((result["entries"][1]["path"], result["entries"][1]["gitPath"]), ("foo.py", True))

    def test_loaded_normalized_exact_search_keeps_session_probe_when_search_misses(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "./foo.py",
                "fileSearchLoadedQuery": "./foo.py",
                "fileSearchResults": [],
            }
        )
        self.assertTrue(result["entries"][0].get("pendingSessionPath", False))
        self.assertEqual((result["entries"][0]["path"], result["entries"][0]["gitPath"]), ("foo.py", False))
        self.assertEqual((result["entries"][1]["path"], result["entries"][1]["gitPath"]), ("foo.py", True))

    def test_error_normalized_exact_search_keeps_session_probe(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "./foo.py",
                "fileSearchErrorQuery": "./foo.py",
            }
        )
        self.assertTrue(result["entries"][0].get("pendingSessionPath", False))
        self.assertEqual((result["entries"][0]["path"], result["entries"][0]["gitPath"]), ("foo.py", False))
        self.assertEqual((result["entries"][1]["path"], result["entries"][1]["gitPath"]), ("foo.py", True))

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
        self.assertEqual(result["applyModeCount"], 5)
        self.assertEqual(
            result["first"],
            [
                {"path": "changed.py", "gitPath": True, "source": "changed"},
                {"path": "recent.txt", "gitPath": False, "source": "recent"},
            ],
        )
        self.assertEqual(result["second"], result["first"])
        self.assertEqual(result["third"], result["first"])
        self.assertEqual(result["cacheSize"], 1)
        self.assertEqual(result["sources"], ["changed", "recent"])
        self.assertEqual(result["literalFiles"], ["/repo/trail.md ", "/repo/new\n.md"])
        self.assertEqual(result["literalRel"], "trail.md ")
        self.assertEqual(result["literalSuffix"], "/repo/trail.md ")
        self.assertEqual(
            result["literalCandidates"],
            [
                {"path": "trail.md ", "gitPath": True, "source": "changed"},
                {"path": "new\n.md", "gitPath": False, "source": "recent"},
            ],
        )
        self.assertIn("/repo/trail.md ", result["literalCacheKey"])
        self.assertTrue(result["firstFresh"])
        self.assertFalse(result["secondFresh"])
        self.assertTrue(result["thirdFresh"])

    def test_resolve_file_open_mode_requires_fresh_changed_metadata_for_diff(self) -> None:
        result = eval_resolve_file_open_mode_cases()
        self.assertEqual(result["freshChanged"], "diff")
        self.assertEqual(result["cachedChanged"], "file")
        self.assertEqual(result["explicitCachedChanged"], "file")
        self.assertEqual(result["freshExplicitUnchanged"], "file")
        self.assertEqual(result["freshExplicitGitFalse"], "file")
        self.assertIsNotNone(result["gitFalseInspect"])
        self.assertEqual(result["staleMarkdownPreview"], "preview")
        self.assertEqual(result["freshChangedNondiffable"], "file")
        self.assertEqual(result["staleRememberedGitPath"], "file")
        self.assertEqual(result["deletedChanged"], "diff")

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
        self.assertIn("openFilePathWithResolvedMode(path, { line: null, changed: Boolean(entry.changed), gitPath: Boolean(entry.gitPath) })", source)
        self.assertIn("openFilePathWithResolvedMode(active.path, { line: null, changed: Boolean(active.changed), gitPath: Boolean(active.gitPath) })", source)
        self.assertIn("compareFilePickerEntries", source)
        self.assertIn("prependPendingSessionPathEntry(localFilePickerSearchEntries(query), query)", source)
        self.assertIn("function filePickerCandidateScore(path, query)", source)
        self.assertIn("applyFileMode();\n            renderFilePickerMenu();", source)
        self.assertIn("const diffable = canToggleMode && activeFileGitPath && fileCandidateGitStateFresh", source)
        self.assertIn("const canUseDiffView = request.gitPath && fileCandidateGitStateFresh", source)
        self.assertIn("fileCandidateKeyForEntry(entry)", source)
        self.assertIn("fileEntryMap.has(entry.key)", source)
        self.assertIn(".fileMenuSection", css)


if __name__ == "__main__":
    unittest.main()
