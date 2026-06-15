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


def eval_file_picker_match_range_helpers() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = "\n".join(
        js_function(source, name)
        for name in [
            "normalizeDraftFilePath",
            "filePickerFoldedSearchText",
            "filePickerOriginalRangeForFolded",
            "filePickerMatchRanges",
            "filePickerMatchRangesForQuery",
        ]
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        function slices(text, ranges) {{ return ranges.map(([start, end]) => text.slice(start, end)); }}
        const turkishText = "İfoo.py";
        const emojiText = "a😀-b.txt";
        const turkish = ctx.filePickerMatchRangesForQuery(turkishText, "foo");
        const emoji = ctx.filePickerMatchRangesForQuery(emojiText, "😀b");
        process.stdout.write(JSON.stringify({{
          exact: ctx.filePickerMatchRangesForQuery("src/foo_bar.py", "foo"),
          fuzzy: ctx.filePickerMatchRangesForQuery("src/foo_bar.py", "fb"),
          normalized: ctx.filePickerMatchRangesForQuery("foo.py", "./foo.py"),
          none: ctx.filePickerMatchRangesForQuery("src/foo.py", "zz"),
          merged: ctx.filePickerMatchRangesForQuery("src/foo.py", "src foo"),
          turkish,
          turkishSlices: slices(turkishText, turkish),
          emoji,
          emojiSlices: slices(emojiText, emoji),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_inline_file_ref_inspection_cases() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = "\n".join(
        js_function(source, name)
        for name in [
            "fileRefValidationKey",
            "normalizeFileRefCandidate",
            "exactBareFileRefMatches",
            "fileRefEntriesMayReferToSamePath",
            "searchBareFileRefCandidates",
            "inspectFileRefCandidate",
            "equivalentFileRefInspection",
            "inspectFileRefPath",
        ]
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          selected: "s1",
          fileRefValidationCache: new Map(),
          fileRefValidationPending: new Map(),
          fileRefSearchCache: new Map(),
          knownCandidates: [],
          searchByQuery: {{}},
          truncatedByQuery: {{}},
          failQueries: {{}},
          inspectFailures: {{}},
          inspectResolvedPaths: {{}},
          apiCalls: [],
          inspectBodies: [],
          getKnownFileRefCandidates: async () => ctx.knownCandidates,
          api: async (url, options = {{}}) => {{
            ctx.apiCalls.push(String(url));
            if (String(url).includes("/file/search")) {{
              const parsed = new URL(String(url), "http://localhost");
              const query = parsed.searchParams.get("q") || "";
              if (ctx.failQueries[query]) throw new Error("search failed");
              const matches = (ctx.searchByQuery[query] || []).map((path) => ({{ path }}));
              return {{ matches, truncated: Boolean(ctx.truncatedByQuery[query]) }};
            }}
            if (String(url) === "/api/files/inspect") {{
              ctx.inspectBodies.push(options.body || {{}});
              const bodyPath = options && options.body ? String(options.body.path || "") : "";
              if ((ctx.inspectFailures[bodyPath] || 0) > 0) {{
                ctx.inspectFailures[bodyPath] -= 1;
                throw new Error("missing");
              }}
              const identityKey = `${{options.body.git_path ? "git" : "session"}}:${{bodyPath}}`;
              return {{ kind: "text", path: ctx.inspectResolvedPaths[identityKey] || options.body.path }};
            }}
            throw new Error("unexpected api call " + url);
          }},
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        (async () => {{
          ctx.knownCandidates = ["src/foo.py", "tests/foo.py"];
          const knownDuplicate = await ctx.inspectFileRefPath("foo.py");
          const callsAfterKnown = ctx.apiCalls.slice();

          ctx.knownCandidates = [];
          ctx.searchByQuery["bar.py"] = ["src/bar.py", "tests/bar.py"];
          const searchedDuplicate = await ctx.inspectFileRefPath("bar.py");

          ctx.knownCandidates = [];
          ctx.searchByQuery["only.py"] = ["src/only.py"];
          const searchedUnique = await ctx.inspectFileRefPath("only.py");

          ctx.knownCandidates = [{{ path: "sub/a.txt", gitPath: true }}];
          ctx.searchByQuery["a.txt"] = ["a.txt"];
          ctx.inspectResolvedPaths["git:sub/a.txt"] = "/repo/sub/a.txt";
          ctx.inspectResolvedPaths["session:a.txt"] = "/repo/sub/a.txt";
          const samePhysical = await ctx.inspectFileRefPath("a.txt");

          ctx.knownCandidates = [];
          ctx.searchByQuery["wide.py"] = ["src/wide.py"];
          ctx.truncatedByQuery["wide.py"] = true;
          const truncatedUnique = await ctx.inspectFileRefPath("wide.py");

          ctx.knownCandidates = [];
          ctx.searchByQuery["emptywide.py"] = [];
          ctx.truncatedByQuery["emptywide.py"] = true;
          const truncatedEmpty = await ctx.inspectFileRefPath("emptywide.py");

          ctx.knownCandidates = [];
          ctx.failQueries["retry.py"] = true;
          const failedSearch = await ctx.inspectFileRefPath("retry.py");
          ctx.failQueries["retry.py"] = false;
          ctx.searchByQuery["retry.py"] = ["src/retry.py"];
          const retriedSearch = await ctx.inspectFileRefPath("retry.py");

          ctx.knownCandidates = [];
          ctx.searchByQuery["late.py"] = [];
          ctx.inspectFailures["late.py"] = 1;
          const missingFirst = await ctx.inspectFileRefPath("late.py");
          const inspectCallsAfterMissingFirst = ctx.inspectBodies.filter((body) => body.path === "late.py").length;
          const missingSecond = await ctx.inspectFileRefPath("late.py");
          const inspectCallsAfterMissingSecond = ctx.inspectBodies.filter((body) => body.path === "late.py").length;

          process.stdout.write(JSON.stringify({{
            knownDuplicate,
            callsAfterKnown,
            searchedDuplicate,
            searchedUnique,
            samePhysical,
            truncatedUnique,
            truncatedEmpty,
            failedSearch,
            retriedSearch,
            missingFirst,
            missingSecond,
            inspectCallsAfterMissingFirst,
            inspectCallsAfterMissingSecond,
            apiCalls: ctx.apiCalls,
            inspectBodies: ctx.inspectBodies,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_file_picker_identity_helpers() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    snippet = "\n".join(
        js_function(source, name)
        for name in ["filePickerIdentityHint", "filePickerTitle", "duplicateFilePickerPaths"]
    )
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(snippet)}, ctx);
        const duplicateEntries = [
          {{ path: "foo.py", gitPath: false, changed: false }},
          {{ path: "foo.py", gitPath: true, changed: true, additions: 1, deletions: 0 }},
          {{ path: "bar.py", gitPath: true, changed: true }},
          {{ path: "draft.py", createNew: true }},
        ];
        const duplicatePaths = ctx.duplicateFilePickerPaths(duplicateEntries);
        const sessionHint = ctx.filePickerIdentityHint(duplicateEntries[0], duplicatePaths);
        const gitHint = ctx.filePickerIdentityHint(duplicateEntries[1], duplicatePaths);
        const changedOnlySectionHint = ctx.filePickerIdentityHint(duplicateEntries[2], duplicatePaths, {{ showSourceSections: true }});
        const changedOnlySearchHint = ctx.filePickerIdentityHint(duplicateEntries[2], duplicatePaths, {{ showSourceSections: false }});
        const createHint = ctx.filePickerIdentityHint(duplicateEntries[3], duplicatePaths);
        const pendingHint = ctx.filePickerIdentityHint({{ path: "foo.py", gitPath: false, pendingSessionPath: true }}, new Set(["foo.py"]));
        process.stdout.write(JSON.stringify({{
          duplicatePaths: Array.from(duplicatePaths),
          sessionHint,
          gitHint,
          changedOnlySectionHint,
          changedOnlySearchHint,
          createHint,
          pendingHint,
          title: ctx.filePickerTitle(duplicateEntries[1], gitHint),
          plainTitle: ctx.filePickerTitle({{ path: "plain.txt" }}, ""),
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


def eval_file_candidates_while_changed_files_pending() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    names = [
        "listFromFilesField",
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
          selected: "s1",
          sessionIndex: new Map([["s1", {{ cwd: "/repo", files: ["/repo/recent.txt"] }}]]),
          chatInner: {{ querySelectorAll() {{ return []; }} }},
          renderCount: 0,
          applyModeCount: 0,
          apiCalls: [],
          resolveChangedFiles: null,
        }};
        vm.createContext(ctx);
        vm.runInContext(`
          class Element {{
            constructor(attrs) {{ this.attrs = attrs || {{}}; }}
            getAttribute(name) {{ return this.attrs[name] || ""; }}
          }}
          const messageNodes = [new Element({{ "data-file-path": "/repo/src/app.py", "data-file-kind": "text" }})];
          chatInner.querySelectorAll = () => messageNodes;
          async function api(path) {{
            apiCalls.push(String(path));
            return await new Promise((resolve) => {{ resolveChangedFiles = resolve; }});
          }}
          function applyFileMode() {{ applyModeCount += 1; }}
          function renderFilePickerMenu() {{ renderCount += 1; }}
          function isFileViewerSessionCurrent() {{ return true; }}
          function blockUnavailableFileAction() {{ return false; }}
        ` + {json.dumps(snippet)}, ctx);
        (async () => {{
          const task = ctx.refreshFileCandidates();
          const interim = {{
            entries: ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }})),
            fresh: ctx.fileCandidateGitStateFresh,
            renderCount: ctx.renderCount,
            applyModeCount: ctx.applyModeCount,
          }};
          ctx.resolveChangedFiles({{ entries: [{{ path: "changed.py", additions: 1, deletions: 0 }}] }});
          await task;
          process.stdout.write(JSON.stringify({{
            interim,
            finalEntries: ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }})),
            finalFresh: ctx.fileCandidateGitStateFresh,
            cacheSize: ctx.fileCandidateCache.size,
            apiCalls: ctx.apiCalls,
            renderCount: ctx.renderCount,
            applyModeCount: ctx.applyModeCount,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_file_candidates_after_changed_files_failure() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    names = [
        "listFromFilesField",
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
          selected: "s1",
          sessionIndex: new Map([["s1", {{ cwd: "/repo", files: ["/repo/recent.txt"] }}]]),
          chatInner: {{ querySelectorAll() {{ return []; }} }},
          renderCount: 0,
          applyModeCount: 0,
          apiCalls: [],
        }};
        vm.createContext(ctx);
        vm.runInContext(`
          class Element {{
            constructor(attrs) {{ this.attrs = attrs || {{}}; }}
            getAttribute(name) {{ return this.attrs[name] || ""; }}
          }}
          const messageNodes = [
            new Element({{ "data-file-path": "/repo/src/app.py", "data-file-kind": "text" }}),
            new Element({{ "data-file-path": "/repo/docs", "data-file-kind": "directory" }}),
          ];
          chatInner.querySelectorAll = () => messageNodes;
          async function api(path) {{
            apiCalls.push(String(path));
            throw new Error("not a git repository");
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
          await ctx.refreshFileCandidates();
          process.stdout.write(JSON.stringify({{
            entries: ctx.currentFileCandidateEntries().map((entry) => ({{ path: entry.path, gitPath: entry.gitPath, source: entry.source }})),
            fresh: ctx.fileCandidateGitStateFresh,
            cacheSize: ctx.fileCandidateCache.size,
            apiCalls: ctx.apiCalls,
            renderCount: ctx.renderCount,
            applyModeCount: ctx.applyModeCount,
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

    def test_same_display_session_identity_beats_git_even_with_lower_score(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "foo.py",
                "fileSearchLoadedQuery": "foo.py",
                "fileSearchResults": [{"path": "foo.py", "score": 1}],
            }
        )
        self.assertEqual((result["entries"][0]["path"], result["entries"][0]["gitPath"]), ("foo.py", False))
        self.assertEqual((result["entries"][1]["path"], result["entries"][1]["gitPath"]), ("foo.py", True))

    def test_same_display_score_normalization_remains_transitive(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                    {"path": "foo.py.bak", "changed": False, "source": "recent", "gitPath": False},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "foo.py",
                "fileSearchLoadedQuery": "foo.py",
                "fileSearchResults": [{"path": "foo.py", "score": 1}],
            }
        )
        self.assertEqual(
            [(entry["path"], entry["gitPath"]) for entry in result["entries"][:3]],
            [("foo.py", False), ("foo.py", True), ("foo.py.bak", False)],
        )

    def test_unrelated_high_score_still_beats_same_display_group(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": "foo.py", "changed": True, "source": "changed", "additions": 1, "deletions": 0},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "foo.py",
                "fileSearchLoadedQuery": "foo.py",
                "fileSearchResults": [
                    {"path": "foo.py", "score": 1},
                    {"path": "bar.py", "score": 13000},
                ],
            }
        )
        self.assertEqual(result["entries"][0]["path"], "bar.py")
        self.assertEqual((result["entries"][1]["path"], result["entries"][1]["gitPath"]), ("foo.py", False))
        self.assertEqual((result["entries"][2]["path"], result["entries"][2]["gitPath"]), ("foo.py", True))

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

    def test_file_candidates_show_fallback_while_changed_files_pending(self) -> None:
        result = eval_file_candidates_while_changed_files_pending()
        self.assertEqual(
            result["interim"]["entries"],
            [
                {"path": "src/app.py", "gitPath": False, "source": "mentioned"},
                {"path": "recent.txt", "gitPath": False, "source": "recent"},
            ],
        )
        self.assertFalse(result["interim"]["fresh"])
        self.assertEqual(result["interim"]["renderCount"], 1)
        self.assertEqual(result["interim"]["applyModeCount"], 2)
        self.assertEqual(
            result["finalEntries"],
            [
                {"path": "changed.py", "gitPath": True, "source": "changed"},
                {"path": "src/app.py", "gitPath": False, "source": "mentioned"},
                {"path": "recent.txt", "gitPath": False, "source": "recent"},
            ],
        )
        self.assertTrue(result["finalFresh"])
        self.assertEqual(result["cacheSize"], 1)
        self.assertEqual(result["renderCount"], 2)
        self.assertEqual(result["applyModeCount"], 3)

    def test_file_candidates_survive_changed_files_failure(self) -> None:
        result = eval_file_candidates_after_changed_files_failure()
        self.assertEqual(
            result["entries"],
            [
                {"path": "src/app.py", "gitPath": False, "source": "mentioned"},
                {"path": "recent.txt", "gitPath": False, "source": "recent"},
            ],
        )
        self.assertFalse(result["fresh"])
        self.assertEqual(result["cacheSize"], 0)
        self.assertEqual(result["apiCalls"], ["/api/sessions/s1/git/changed_files"])
        self.assertEqual(result["renderCount"], 1)
        self.assertEqual(result["applyModeCount"], 2)

    def test_file_candidate_cache_reuses_same_session_key(self) -> None:
        result = eval_file_candidate_cache_helpers()
        self.assertEqual(result["apiCalls"], 2)
        self.assertEqual(result["renderCount"], 5)
        self.assertEqual(result["applyModeCount"], 7)
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

    def test_inline_file_refs_use_project_search_when_known_candidates_are_inconclusive(self) -> None:
        result = eval_inline_file_ref_inspection_cases()
        self.assertFalse(result["knownDuplicate"]["ok"])
        self.assertTrue(result["knownDuplicate"]["ambiguous"])
        self.assertEqual(result["callsAfterKnown"], [])
        self.assertFalse(result["searchedDuplicate"]["ok"])
        self.assertTrue(result["searchedDuplicate"]["ambiguous"])
        self.assertTrue(result["searchedUnique"]["ok"])
        self.assertEqual(result["searchedUnique"]["inspectPath"], "src/only.py")
        self.assertTrue(result["samePhysical"]["ok"])
        self.assertFalse(result["samePhysical"].get("ambiguous", False))
        self.assertEqual(result["samePhysical"]["inspectPath"], "a.txt")
        self.assertEqual(result["samePhysical"]["resolvedPath"], "/repo/sub/a.txt")
        self.assertFalse(result["truncatedUnique"]["ok"])
        self.assertTrue(result["truncatedUnique"]["ambiguous"])
        self.assertFalse(result["truncatedEmpty"]["ok"])
        self.assertTrue(result["truncatedEmpty"]["ambiguous"])
        self.assertFalse(result["failedSearch"]["ok"])
        self.assertTrue(result["failedSearch"]["ambiguous"])
        self.assertTrue(result["retriedSearch"]["ok"])
        self.assertEqual(result["retriedSearch"]["inspectPath"], "src/retry.py")
        self.assertFalse(result["missingFirst"]["ok"])
        self.assertTrue(result["missingSecond"]["ok"])
        self.assertEqual(result["inspectCallsAfterMissingFirst"], 1)
        self.assertEqual(result["inspectCallsAfterMissingSecond"], 2)
        self.assertIn("/api/sessions/s1/file/search?q=bar.py&limit=80", result["apiCalls"])
        self.assertEqual(result["apiCalls"].count("/api/sessions/s1/file/search?q=retry.py&limit=80"), 2)
        self.assertIn({"session_id": "s1", "path": "src/only.py"}, result["inspectBodies"])
        self.assertIn({"session_id": "s1", "path": "sub/a.txt", "git_path": True}, result["inspectBodies"])
        self.assertIn({"session_id": "s1", "path": "a.txt"}, result["inspectBodies"])
        self.assertIn({"session_id": "s1", "path": "src/retry.py"}, result["inspectBodies"])

    def test_file_picker_identity_hints_only_explain_ambiguous_resolution(self) -> None:
        result = eval_file_picker_identity_helpers()
        self.assertEqual(result["duplicatePaths"], ["foo.py"])
        self.assertEqual(result["sessionHint"], "current folder")
        self.assertEqual(result["gitHint"], "git root · changed")
        self.assertEqual(result["changedOnlySectionHint"], "")
        self.assertEqual(result["changedOnlySearchHint"], "git root · changed")
        self.assertEqual(result["pendingHint"], "current folder")
        self.assertEqual(result["createHint"], "")
        self.assertEqual(result["title"], "foo.py — git root · changed")
        self.assertEqual(result["plainTitle"], "plain.txt")

    def test_file_picker_match_ranges_support_exact_fuzzy_and_normalized_queries(self) -> None:
        result = eval_file_picker_match_range_helpers()
        self.assertEqual(result["exact"], [[4, 7]])
        self.assertEqual(result["fuzzy"], [[4, 5], [8, 9]])
        self.assertEqual(result["normalized"], [[0, 6]])
        self.assertEqual(result["none"], [])
        self.assertEqual(result["merged"], [[0, 3], [4, 7]])
        self.assertEqual(result["turkishSlices"], ["foo"])
        self.assertEqual(result["emojiSlices"], ["😀", "b"])

    def test_file_picker_highlights_matches_without_rewriting_path_identity(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = (APP_JS.parent / "app.css").read_text(encoding="utf-8")
        start = source.index("function appendHighlightedFileMenuPath(parent, text, query) {")
        end = source.index("function resetFileSearchState()", start)
        block = source[start:end]
        self.assertIn('span.appendChild(document.createTextNode(value.slice(cursor, start)));', block)
        self.assertIn('span.appendChild(el("mark", { class: "fileMenuMatch", text: value.slice(start, end) }));', block)
        self.assertIn("parent.appendChild(span);", block)
        self.assertNotIn("innerHTML", block)
        self.assertIn("appendHighlightedFileMenuPath(btn, path, query);", source)
        self.assertIn("appendHighlightedFileMenuPath(btn, `Create new file: ${path}`, query);", source)
        self.assertIn("title: filePickerTitle(entry, identityHint)", source)
        self.assertIn(".fileMenuMatch", css)

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
        self.assertIn("openFilePathWithResolvedMode(path, { line: filePickerSelectionLine(), changed: Boolean(entry.changed), gitPath: Boolean(entry.gitPath) })", source)
        self.assertIn("openFilePathWithResolvedMode(active.path, { line: filePickerSelectionLine(), changed: Boolean(active.changed), gitPath: Boolean(active.gitPath) })", source)
        self.assertIn("compareFilePickerEntries", source)
        self.assertIn("prependPendingSessionPathEntry(localFilePickerSearchEntries(query), query)", source)
        self.assertIn("function filePickerCandidateScore(path, query)", source)
        self.assertIn("applyFileMode();\n            renderFilePickerMenu();", source)
        self.assertIn("const diffable = canToggleMode && activeFileGitPath && fileCandidateGitStateFresh", source)
        self.assertIn("const canUseDiffView = request.gitPath && fileCandidateGitStateFresh", source)
        self.assertIn("fileCandidateKeyForEntry(entry)", source)
        self.assertIn("fileEntryMap.has(entry.key)", source)
        self.assertIn(".fileMenuSection", css)
        self.assertIn("function filePickerIdentityHint(entry, duplicatePaths, options)", source)
        self.assertIn("function duplicateFilePickerPaths(entries)", source)
        self.assertNotIn('"aria-label": filePickerTitle(entry, identityHint)', source)
        self.assertIn("title: filePickerTitle(entry, identityHint)", source)
        self.assertIn("fileMenuHint fileMenuIdentity", source)
        self.assertIn(".fileMenuIdentity", css)


if __name__ == "__main__":
    unittest.main()
