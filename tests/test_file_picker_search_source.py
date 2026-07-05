import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_FILE_PICKER_JS = ROOT / "codoxear" / "static" / "app_file_picker.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"


def js_candidate_controller_fixture() -> str:
    return r'''
          function __candidateKey(path, gitPath = false, apiPath = "") {
            const token = normalizeFileApiPath(apiPath);
            const identity = token || String(path ?? "");
            return `${gitPath ? "git" : "session"}\u0000${identity}`;
          }
          function __candidateClone(entry) {
            if (!entry || typeof entry.path !== "string" || entry.path === "") return null;
            const source = normalizeFileCandidateSource(entry.source);
            const gitPath = entry.gitPath === undefined ? Boolean(entry.changed && source === "changed") : Boolean(entry.gitPath);
            const apiPath = normalizeFileApiPath(entry.apiPath || entry.api_path);
            const untracked = Boolean(entry.untracked);
            const oldPath = typeof entry.oldPath === "string" ? entry.oldPath : "";
            const rename = Boolean(entry.rename || oldPath);
            return { path: entry.path, apiPath, gitPath, key: __candidateKey(entry.path, gitPath, apiPath), additions: entry.additions ?? null, deletions: entry.deletions ?? null, changed: untracked ? false : Boolean(entry.changed), untracked, rename, oldPath, source };
          }
          fileViewerController = {
            beginFileCandidateRefresh() { fileCandidateRequestSeq += 1; return fileCandidateRequestSeq; },
            isCurrentFileCandidateRefresh(requestSeq) { return requestSeq === fileCandidateRequestSeq; },
            fileCandidateKey: __candidateKey,
            fileCandidateKeyForEntry(entry) { return __candidateKey(entry && entry.path, Boolean(entry && entry.gitPath), normalizeFileApiPath(entry && entry.apiPath)); },
            cloneFileCandidateEntry: __candidateClone,
            applyFileCandidateEntries(entries) {
              fileCandidateList = [];
              fileEntryMap = new Map();
              for (const raw of Array.isArray(entries) ? entries : []) {
                const entry = __candidateClone(raw);
                if (!entry || fileEntryMap.has(entry.key)) continue;
                fileCandidateList.push(entry.key);
                fileEntryMap.set(entry.key, entry);
              }
            },
            currentFileCandidateKeys() { return fileCandidateList.slice(); },
            currentFileCandidateEntries() { return fileCandidateList.map((key) => __candidateClone(fileEntryMap.get(key))).filter(Boolean); },
            fileEntryForKey(key) { return __candidateClone(fileEntryMap.get(String(key || ""))); },
            fileEntryForPath(path, gitPath = false, apiPath = "") {
              const token = normalizeFileApiPath(apiPath);
              const preferred = fileEntryMap.get(__candidateKey(path, gitPath, token));
              if (preferred) return __candidateClone(preferred);
              const fallback = fileEntryMap.get(__candidateKey(path, gitPath));
              if (fallback && (!token || normalizeFileApiPath(fallback.apiPath) === token)) return __candidateClone(fallback);
              for (const key of fileCandidateList) {
                const entry = fileEntryMap.get(key);
                if (!entry || entry.path !== path || Boolean(entry.gitPath) !== Boolean(gitPath)) continue;
                if (!token || normalizeFileApiPath(entry.apiPath) === token) return __candidateClone(entry);
              }
              return null;
            },
            fileApiPathForPath(path, apiPath = "") {
              const existing = normalizeFileApiPath(apiPath);
              if (existing) return existing;
              const entry = this.fileEntryForPath(path, true) || this.fileEntryForPath(path, false);
              return normalizeFileApiPath(entry && entry.apiPath);
            },
            activeFileEntry() { return null; },
            isGitFileCandidatePath(path, changed = null, gitPath = null, apiPath = "") {
              if (gitPath !== null && gitPath !== undefined) return Boolean(gitPath);
              if (changed !== null && changed !== undefined) return Boolean(changed);
              const gitEntry = this.fileEntryForPath(path, true, apiPath);
              if (gitEntry) return true;
              const sessionEntry = this.fileEntryForPath(path, false);
              return Boolean(sessionEntry && sessionEntry.gitPath);
            },
            currentFileCandidateGitStateFresh() { return fileCandidateGitStateFresh; },
            setFileCandidateGitStateFresh(fresh) { fileCandidateGitStateFresh = Boolean(fresh); return fileCandidateGitStateFresh; },
            currentFileCandidateGitStateMessage() { return fileCandidateGitStateMessage; },
            setFileCandidateGitStateMessage(message) { fileCandidateGitStateMessage = String(message || ""); return fileCandidateGitStateMessage; },
            rememberFileCandidateCache(sid, key, now = Date.now()) { if (!sid || !key) return false; fileCandidateCache.set(sid, { key, ts: Number(now || 0), entries: this.currentFileCandidateEntries() }); return true; },
            fileCandidateCacheEntry(sid) { const cached = fileCandidateCache.get(String(sid || "")); return cached ? { key: cached.key, ts: cached.ts, entries: (cached.entries || []).map(__candidateClone).filter(Boolean) } : null; },
            deleteFileCandidateCache(sid) { return fileCandidateCache.delete(String(sid || "")); },
            fileCandidateCacheSize() { return fileCandidateCache.size; },
            applyFileCandidateRefreshEntries(entries, { gitStateFresh = false, gitStateMessage = "" } = {}) {
              this.applyFileCandidateEntries(entries);
              this.setFileCandidateGitStateFresh(gitStateFresh);
              fileCandidateGitStateMessage = gitStateFresh ? "" : String(gitStateMessage || "");
              applyFileMode();
              return true;
            },
            clearFileCandidateRefreshEntries() { return this.applyFileCandidateRefreshEntries([], { gitStateFresh: false }); },
            applyFreshFileCandidateCache(sid, key, { now = Date.now(), ttl = 0 } = {}) {
              const cached = this.fileCandidateCacheEntry(sid);
              if (!cached || cached.key !== key) return false;
              const age = Number(now || 0) - Number(cached.ts || 0);
              if (!(age >= 0 && age < Number(ttl || 0))) return false;
              return this.applyFileCandidateRefreshEntries(cached.entries, { gitStateFresh: false });
            },
            upsertFileEntry(entry) {
              const merged = __candidateClone(entry);
              if (!merged) return false;
              const current = fileEntryMap.get(merged.key);
              if (current && !merged.source) merged.source = normalizeFileCandidateSource(current.source);
              if (!fileEntryMap.has(merged.key)) fileCandidateList.push(merged.key);
              fileEntryMap.set(merged.key, merged);
              return true;
            },
            pickerEntryForKey(key, { score = 0 } = {}) { const entry = this.fileEntryForKey(key); return entry ? { ...entry, added: true, score } : null; },
            pickerEntryForPath(path, { score = 0, gitPath = false, apiPath = "" } = {}) { const token = normalizeFileApiPath(apiPath); const key = __candidateKey(path, gitPath, token); const existing = fileEntryMap.get(key); const entry = __candidateClone({ ...(existing || { path, gitPath, additions: null, deletions: null, changed: false, source: "" }), apiPath: token }); return entry ? { ...entry, added: Boolean(existing), score } : null; },
          };
    '''


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
    wrapper_names = [
        "normalizeFileApiPath",
        "fileSearchScore",
        "normalizeDraftFilePath",
        "filePickerFoldedSearchText",
        "filePickerOriginalRangeForFolded",
        "filePickerMatchRanges",
        "filePickerMatchRangesForQuery",
        "filePickerCandidateScore",
        "compareFilePickerEntries",
        "normalizeFileCandidateSource",
    ]
    snippet = "\n".join(js_function(source, name) for name in wrapper_names)
    snippet_with_helpers = "const codoxearFileHelpers = window.CodoxearFileHelpers;\nconst codoxearFilePicker = window.CodoxearFilePicker;\n" + snippet
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    file_helpers_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    file_picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const state = {json.dumps(state)};
        const ctx = {{
          window: {{}},
          fileCandidateList: [],
          fileEntryMap: new Map(),
          fileCandidateGitStateFresh: false,
          fileCandidateGitStateMessage: "",
          fileCandidateCache: new Map(),
          fileCandidateRequestSeq: 0,
          activeFileDraft: Boolean(state.activeFileDraft),
          currentActiveFileDraft: () => Boolean(state.activeFileDraft),
          activeFilePathValue: () => state.activeFilePath || "",
          filePickerSearchActive: Boolean(state.filePickerSearchActive),
          filePickerInput: {{ value: state.filePickerInputValue || "" }},
          filePickerSuppressDraftQuery: state.filePickerSuppressDraftQuery || "",
          normalizeLineNumber: (value) => {{
            const n = Number(value);
            return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
          }},
          filePickerSearchState: {{
            snapshot: () => ({{
              results: state.fileSearchResults || [],
              loadedQuery: state.fileSearchLoadedQuery || "",
              pendingQuery: state.fileSearchPendingQuery || "",
              errorQuery: state.fileSearchErrorQuery || "",
              error: state.fileSearchError || "",
              truncatedQuery: state.fileSearchTruncatedQuery || "",
              sessionId: state.fileSearchSessionId || "",
            }}),
          }},
          baseName: (path) => String(path || "").split("/").pop() || "",
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(file_helpers_source)}, ctx);
        vm.runInContext({json.dumps(file_picker_source)}, ctx);
        vm.runInContext(`
          globalThis.filePickerMenuState = window.CodoxearFilePicker.createMenuState({{ normalizeLineNumber }});
          if (filePickerSearchActive) filePickerMenuState.handleInput(filePickerInput.value);
          if (filePickerSuppressDraftQuery) filePickerMenuState.openSearchQuery(filePickerSuppressDraftQuery, {{ suppressDraft: true }});
        `, ctx);
        vm.runInContext({json.dumps(snippet_with_helpers)}, ctx);
        vm.runInContext({json.dumps(js_candidate_controller_fixture())}, ctx);
        vm.runInContext(`
          globalThis.filePickerEntryRuntime = window.CodoxearFilePicker.createEntryRuntime({{
            menuState: filePickerMenuState,
            inputValue: () => filePickerInput.value,
            candidateKeys: () => fileViewerController.currentFileCandidateKeys(),
            entryForKey: (key) => fileViewerController.fileEntryForKey(key),
            pickerEntryForKey: (key, options) => fileViewerController.pickerEntryForKey(key, options),
            pickerEntryForPath: (path, options) => fileViewerController.pickerEntryForPath(path, options),
            keyForPath: (path, gitPath, apiPath) => fileViewerController.fileCandidateKey(path, gitPath, apiPath),
            activeFileDraft: () => currentActiveFileDraft(),
            activeFilePath: () => activeFilePathValue(),
            searchSnapshot: () => filePickerSearchState.snapshot(),
            normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          }});
          globalThis.__test_file_picker_search = {{
            applyFileCandidateEntries: (entries) => fileViewerController.applyFileCandidateEntries(entries),
            visibleFilePickerEntries: () => filePickerEntryRuntime.visibleEntries(),
            localFilePickerSearchEntries: (query) => window.CodoxearFilePicker.localFilePickerSearchEntries(filePickerEntryRuntime.entryContext(query), query),
          }};
        `, ctx);
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
    snippet_with_helpers = "const codoxearFileHelpers = window.CodoxearFileHelpers;\n" + snippet
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    file_helpers_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(file_helpers_source)}, ctx);
        vm.runInContext({json.dumps(snippet_with_helpers)}, ctx);
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
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          window: {{}},
          selected: "s1",
          knownCandidates: [],
          changedEntries: [],
          searchByQuery: {{}},
          truncatedByQuery: {{}},
          failQueries: {{}},
          inspectFailures: {{}},
          inspectResolvedPaths: {{}},
          apiCalls: [],
          inspectBodies: [],
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        ctx.runtime = ctx.window.CodoxearFileViewer.createFileReferenceRuntime({{
          selectedSessionId: () => ctx.selected,
          sessionById: () => ({{ files: ctx.knownCandidates }}),
          chatRoot: {{ querySelectorAll: () => [] }},
          ElementCtor: null,
          sessionRelativePath: (rawPath) => String(rawPath || ""),
          listFromFilesField: (value) => Array.isArray(value) ? value.slice() : [],
          listFromFileRecords: (value) => (Array.isArray(value) ? value : []).map((v) => (typeof v === "string" ? {{ path: v, apiPath: "" }} : v && typeof v === "object" ? {{ path: v.path || "", apiPath: v.apiPath || v.api_path || "" }} : null)).filter((r) => r && r.path),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          normalizeLineNumber: (value) => Number(value) || null,
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children, appendChild(child) {{ this.children.push(child); return child; }} }}),
          api: async (url, options = {{}}) => {{
            ctx.apiCalls.push(String(url));
            if (String(url).includes("/git/changed_files")) return {{ entries: ctx.changedEntries }};
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
        }});
        async function inspect(path) {{ return await ctx.runtime.inspectPath(path); }}
        function resetKnown({{ known = [], changed = [] }} = {{}}) {{ ctx.knownCandidates = known; ctx.changedEntries = changed; ctx.runtime.clearDiscoveryCaches(); }}
        (async () => {{
          resetKnown({{ known: ["src/foo.py", "tests/foo.py"] }});
          const knownDuplicate = await inspect("foo.py");
          const callsAfterKnown = ctx.apiCalls.slice();

          resetKnown();
          ctx.searchByQuery["bar.py"] = ["src/bar.py", "tests/bar.py"];
          const searchedDuplicate = await inspect("bar.py");

          resetKnown();
          ctx.searchByQuery["only.py"] = ["src/only.py"];
          const searchedUnique = await inspect("only.py");

          resetKnown({{ changed: [{{ path: "sub/a.txt", api_path: "" }}] }});
          ctx.searchByQuery["a.txt"] = ["a.txt"];
          ctx.inspectResolvedPaths["git:sub/a.txt"] = "/repo/sub/a.txt";
          ctx.inspectResolvedPaths["session:a.txt"] = "/repo/sub/a.txt";
          const samePhysical = await inspect("a.txt");

          resetKnown();
          ctx.searchByQuery["wide.py"] = ["src/wide.py"];
          ctx.truncatedByQuery["wide.py"] = true;
          const truncatedUnique = await inspect("wide.py");

          resetKnown();
          ctx.searchByQuery["emptywide.py"] = [];
          ctx.truncatedByQuery["emptywide.py"] = true;
          const truncatedEmpty = await inspect("emptywide.py");

          resetKnown();
          ctx.failQueries["retry.py"] = true;
          const failedSearch = await inspect("retry.py");
          ctx.failQueries["retry.py"] = false;
          ctx.searchByQuery["retry.py"] = ["src/retry.py"];
          const retriedSearch = await inspect("retry.py");

          resetKnown();
          ctx.searchByQuery["late.py"] = [];
          ctx.inspectFailures["late.py"] = 1;
          const missingFirst = await inspect("late.py");
          const inspectCallsAfterMissingFirst = ctx.inspectBodies.filter((body) => body.path === "late.py").length;
          const missingSecond = await inspect("late.py");
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
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_file_picker_identity_helpers() -> dict:
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        const helpers = ctx.window.CodoxearFileHelpers;
        const duplicateEntries = [
          {{ path: "foo.py", gitPath: false, changed: false }},
          {{ path: "foo.py", gitPath: true, changed: true, additions: 1, deletions: 0 }},
          {{ path: "bar.py", gitPath: true, changed: true }},
          {{ path: "draft.py", createNew: true }},
        ];
        const duplicatePaths = helpers.duplicateFilePickerPaths(duplicateEntries);
        const sessionHint = helpers.filePickerIdentityHint(duplicateEntries[0], duplicatePaths);
        const gitHint = helpers.filePickerIdentityHint(duplicateEntries[1], duplicatePaths);
        const changedOnlySectionHint = helpers.filePickerIdentityHint(duplicateEntries[2], duplicatePaths, {{ showSourceSections: true }});
        const changedOnlySearchHint = helpers.filePickerIdentityHint(duplicateEntries[2], duplicatePaths, {{ showSourceSections: false }});
        const createHint = helpers.filePickerIdentityHint(duplicateEntries[3], duplicatePaths);
        const pendingHint = helpers.filePickerIdentityHint({{ path: "foo.py", gitPath: false, pendingSessionPath: true }}, new Set(["foo.py"]));
        process.stdout.write(JSON.stringify({{
          duplicatePaths: Array.from(duplicatePaths),
          sessionHint,
          gitHint,
          changedOnlySectionHint,
          changedOnlySearchHint,
          createHint,
          pendingHint,
          title: helpers.filePickerTitle(duplicateEntries[1], gitHint),
          plainTitle: helpers.filePickerTitle({{ path: "plain.txt" }}, ""),
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_resolve_file_open_mode_cases() -> dict:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        const state = {{ inspectedKind: "text", inspectCalls: [] }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        const fileStatus = {{ textContent: "", replaceChildren() {{ this.textContent = ""; }} }};
        const fileEditButton = {{ classList: {{ toggle() {{}} }}, setAttribute() {{}}, disabled: false }};
        const controller = ctx.window.CodoxearFileViewer.createFileViewerController({{
          el: (tag, attrs = {{}}, children = []) => ({{ tag, attrs, children }}),
          fileStatus,
          fileEditButton,
          iconSvg: (name) => `icon:${{name}}`,
          currentSessionId: () => "sid-1",
          currentFileSessionId: () => "sid-1",
          normalizeLineNumber: (value) => value == null || value === "" ? null : Number(value),
          normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
          isFileViewerOpen: () => true,
          hideFileUnsavedDialog: () => {{}},
          resetFileSearchState: () => {{}},
          closeFilePickerMenu: () => {{}},
          isTextFileKind: (kind) => kind === "text" || kind === "markdown",
          isDiffableFileKind: (kind) => kind === "text" || kind === "markdown",
          confirmReload: () => true,
          promptUnsavedFileChoice: async () => "cancel",
          restoreFileEditorText: () => {{}},
          hideFileViewer: () => {{}},
          setFilePath: () => {{}},
          resetFileViewerPanel: () => {{}},
          applyFileLoadResult: async () => true,
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          inspectSessionFilePath: async (path, options = {{}}) => {{ state.inspectCalls.push([path, options]); return state.inspectedKind === "missing" ? {{ exists: false }} : {{ exists: true, kind: state.inspectedKind }}; }},
          api: async () => ({{}}),
          focusEditor: () => null,
          disposeOpenRender: () => {{}},
          persistFileViewMode: () => {{}},
          persistFileNonDiffMode: () => {{}},
          isMarkdownPreviewable: () => false,
          updateFileTouchToolbar: () => {{}},
          useTouchFileEditorControls: () => false,
          hasActiveFileCodeEditor: () => false,
          hasBlockingFileEditorModal: () => false,
          isTextEntryTarget: () => false,
          eventTargetElement: (value) => value || null,
          normalizeFileEditorPosition: (_editor, position) => position || null,
          applyFileEditorSelection: () => {{}},
          isCollapsedFileSelection: () => true,
          positionAfterInsertedText: (start, text) => ({{ lineNumber: start.lineNumber, column: start.column + String(text || "").length }}),
          fileEditorEditSupportAvailable: () => false,
          updateFileDiffEditorOptions: () => {{}},
          showFilePasteDialog: () => false,
          hideFilePasteDialog: () => {{}},
          clipboardReadAvailable: () => false,
          readClipboardText: async () => "",
          fileEditorDeleteCommandForKey: () => "",
          isActiveFileEditorInput: () => false,
          getActiveFileSelectionText: () => "",
          copyToClipboard: async () => {{}},
          focusActiveFileCodeEditor: () => null,
          nowMs: () => 0,
          setToast: () => {{}},
          renderMonacoFile: async () => true,
          getFileEditorText: () => "",
          fmtBytes: (value) => String(value),
          applyFileMode: () => {{}},
          rememberOpenedFile: () => {{}},
          historyFileSelectionForSession: () => ({{ path: "", line: null, gitPath: false, apiPath: "" }}),
          renderFilePickerMenu: () => {{}},
        }});
        (async () => {{
          controller.applyFileCandidateEntries([{{ path: "changed.py", gitPath: true, changed: true }}]);
          controller.setFileCandidateGitStateFresh(true);
          state.inspectedKind = "text";
          const freshChanged = await controller.resolveFileOpenMode("changed.py");
          controller.setFileCandidateGitStateFresh(false);
          const cachedChanged = await controller.resolveFileOpenMode("changed.py");
          const explicitCachedChanged = await controller.resolveFileOpenMode("changed.py", {{ changed: true }});
          controller.setFileCandidateGitStateFresh(true);
          const freshExplicitUnchanged = await controller.resolveFileOpenMode("changed.py", {{ changed: false }});
          const freshExplicitGitFalse = await controller.resolveFileOpenMode("changed.py", {{ gitPath: false }});
          controller.setFileCandidateGitStateFresh(false);
          controller.setFileViewMode("preview");
          state.inspectedKind = "markdown";
          const staleMarkdownPreview = await controller.resolveFileOpenMode("README.md", {{ changed: true }});
          controller.setFileCandidateGitStateFresh(true);
          state.inspectedKind = "image";
          const freshChangedNondiffable = await controller.resolveFileOpenMode("image.png", {{ changed: true }});
          controller.setFileCandidateGitStateFresh(true);
          state.inspectedKind = "text";
          const staleRememberedGitPath = await controller.resolveFileOpenMode("stale.py", {{ gitPath: true }});
          state.inspectedKind = "missing";
          const deletedChanged = await controller.resolveFileOpenMode("gone.md", {{ changed: true }});
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
            gitFalseInspect: state.inspectCalls.find((call) => call[0] === "changed.py" && call[1] && call[1].gitPath === false) || null,
          }}));
        }})().catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def file_candidate_refresh_runtime_prelude() -> str:
    viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    file_helpers_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    fixture_source = r'''
      const codoxearFileHelpers = window.CodoxearFileHelpers;
      const codoxearFileViewer = window.CodoxearFileViewer;
      function normalizeFileApiPath(value) { return typeof value === "string" && value !== "" ? value : ""; }
      function normalizeFileCandidateSource(source) { return codoxearFileHelpers.normalizeFileCandidateSource(source); }
      function listFromFilesField(value) { return codoxearFileHelpers.listFromFilesField(value); }
      function listFromFileRecords(value) { return codoxearFileHelpers.listFromFileRecords(value); }
      function stripPathLocationSuffix(rawPath) { return codoxearFileHelpers.stripPathLocationSuffix(rawPath); }
      function sessionRelativePath(rawPath, sidOverride = null) {
        const sid = typeof sidOverride === "string" && sidOverride ? sidOverride : selected;
        const s = sid ? sessionIndex.get(sid) : null;
        if (!s || !s.cwd) return null;
        const abs = stripPathLocationSuffix(rawPath);
        const cwd = String(s.cwd || "").replace(/\/+$/, "");
        if (!abs) return null;
        if (abs === cwd) return ".";
        if (abs.startsWith(cwd + "/")) return abs.slice(cwd.length + 1);
        return null;
      }
      class Element {
        constructor(attrs) { this.attrs = attrs || {}; }
        getAttribute(name) { return this.attrs[name] || ""; }
      }
      function collectMessageFileRefs() {
        if (!selected) return [];
        const out = [];
        const seen = new Set();
        const nodes = Array.from(chatInner.querySelectorAll("[data-file-path]"));
        for (const node of nodes) {
          if (!(node instanceof Element)) continue;
          const kind = String(node.getAttribute("data-file-kind") || "").trim();
          if (kind === "directory") continue;
          const rawAbs = String(node.getAttribute("data-file-path") ?? "");
          const raw = rawAbs;
          if (raw === "") continue;
          const rel = raw.startsWith("/") ? sessionRelativePath(raw) || "" : raw.replace(/^\.?\//, "");
          if (!rel || rel === "." || seen.has(rel)) continue;
          seen.add(rel);
          out.push(rel);
        }
        return out;
      }
      function fileCandidateCacheKey(sid) {
        const s = sid ? sessionIndex.get(sid) : null;
        const filesKey = JSON.stringify(listFromFilesField(s && s.files));
        const refsKey = JSON.stringify(collectMessageFileRefs());
        return `${sid || ""}\u0000${filesKey}\u0000${refsKey}`;
      }
      function applyFileMode() { applyModeCount += 1; }
      function renderFilePickerMenu() { renderCount += 1; }
    '''
    runtime_source = r'''
      const fileCandidateRefreshRuntime = codoxearFileViewer.createFileCandidateRefreshRuntime({
        controller: fileViewerController,
        currentSessionId: () => fileViewerSessionId,
        selectedSessionId: () => selected,
        blockUnavailableFileAction: () => false,
        isSessionCurrent: () => true,
        candidateCacheKey: (sid) => fileCandidateCacheKey(sid),
        ttlMs: FILE_CANDIDATE_CACHE_TTL_MS,
        nowMs: () => now,
        collectMessageFileRefs: () => collectMessageFileRefs(),
        sessionFiles: (sid) => listFromFilesField((sessionIndex.get(sid) || {}).files),
        sessionFileRecords: (sid) => listFromFileRecords((sessionIndex.get(sid) || {}).files),
        sessionRelativePath: (rawPath, sid) => sessionRelativePath(rawPath, sid),
        api: (url) => api(url),
        normalizeFileApiPath: (value) => normalizeFileApiPath(value),
        renderMenu: () => renderFilePickerMenu(),
      });
      async function refreshFileCandidates(options = {}) { return await fileCandidateRefreshRuntime.refresh(options); }
      function currentFileCandidateEntries() { return fileViewerController.currentFileCandidateEntries(); }
    '''
    return textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{
          window: {{}},
          process,
          console,
          fileCandidateList: [],
          fileEntryMap: new Map(),
          fileCandidateGitStateFresh: false,
          fileCandidateGitStateMessage: "",
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
          now: 1000,
        }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(file_helpers_source)}, ctx);
        vm.runInContext({json.dumps(viewer_source)}, ctx);
        vm.runInContext({json.dumps(fixture_source)}, ctx);
        vm.runInContext({json.dumps(js_candidate_controller_fixture())}, ctx);
        vm.runInContext({json.dumps(runtime_source)}, ctx);
        """
    )

def run_file_candidate_refresh_probe(setup_js: str, body_js: str) -> dict:
    js = file_candidate_refresh_runtime_prelude() + textwrap.dedent(
        f"""
        vm.runInContext({json.dumps(setup_js)}, ctx);
        vm.runInContext({json.dumps(body_js)}, ctx);
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def eval_file_candidates_while_changed_files_pending() -> dict:
    setup_js = r'''
      const messageNodes = [new Element({ "data-file-path": "/repo/src/app.py", "data-file-kind": "text" })];
      chatInner.querySelectorAll = () => messageNodes;
      async function api(path) {
        apiCalls.push(String(path));
        return await new Promise((resolve) => { resolveChangedFiles = resolve; });
      }
    '''
    body_js = r'''
      (async () => {
        const task = refreshFileCandidates();
        const interim = {
          entries: currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source })),
          fresh: fileCandidateGitStateFresh,
          renderCount,
          applyModeCount,
        };
        resolveChangedFiles({ entries: [{ path: "changed.py", additions: 1, deletions: 0 }] });
        await task;
        process.stdout.write(JSON.stringify({
          interim,
          finalEntries: currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source })),
          finalFresh: fileCandidateGitStateFresh,
          cacheSize: fileCandidateCache.size,
          apiCalls,
          renderCount,
          applyModeCount,
        }));
      })().catch((err) => { console.error(err && err.stack || err); process.exit(1); });
    '''
    return run_file_candidate_refresh_probe(setup_js, body_js)


def eval_file_candidates_after_changed_files_failure() -> dict:
    setup_js = r'''
      const messageNodes = [
        new Element({ "data-file-path": "/repo/src/app.py", "data-file-kind": "text" }),
        new Element({ "data-file-path": "/repo/docs", "data-file-kind": "directory" }),
      ];
      chatInner.querySelectorAll = () => messageNodes;
      async function api(path) {
        apiCalls.push(String(path));
        throw new Error("not a git repository");
      }
    '''
    body_js = r'''
      (async () => {
        await refreshFileCandidates();
        process.stdout.write(JSON.stringify({
          entries: currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source })),
          fresh: fileCandidateGitStateFresh,
          gitStateMessage: fileCandidateGitStateMessage,
          cacheSize: fileCandidateCache.size,
          apiCalls,
          renderCount,
          applyModeCount,
        }));
      })().catch((err) => { console.error(err && err.stack || err); process.exit(1); });
    '''
    return run_file_candidate_refresh_probe(setup_js, body_js)


def eval_file_candidate_cache_helpers() -> dict:
    setup_js = r'''
      selected = "";
      async function api(path) {
        apiCalls.push(String(path));
        return { entries: [{ path: "changed.py", additions: 1, deletions: 0 }] };
      }
    '''
    body_js = r'''
      (async () => {
        const literalFiles = listFromFilesField(["/repo/trail.md ", "/repo/new\n.md", "", "/repo/trail.md "]);
        const literalRel = sessionRelativePath("/repo/trail.md ", "s1");
        const literalSuffix = stripPathLocationSuffix("/repo/trail.md ");
        fileViewerController.applyFileCandidateEntries([{ path: "trail.md ", changed: true, source: "changed" }, { path: "new\n.md", source: "recent" }]);
        const literalCandidates = currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source }));
        sessionIndex.get("s1").files = literalFiles;
        const literalCacheKey = fileCandidateCacheKey("s1");
        sessionIndex.get("s1").files = ["/repo/recent.txt"];
        fileCandidateList = [];
        fileEntryMap = new Map();
        await refreshFileCandidates();
        const first = currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source }));
        const firstFresh = fileCandidateGitStateFresh;
        now += 1;
        await refreshFileCandidates();
        const second = currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source }));
        const secondFresh = fileCandidateGitStateFresh;
        now += 1;
        await refreshFileCandidates({ force: true });
        const third = currentFileCandidateEntries().map((entry) => ({ path: entry.path, gitPath: entry.gitPath, source: entry.source }));
        const thirdFresh = fileCandidateGitStateFresh;
        process.stdout.write(JSON.stringify({
          apiCalls: apiCalls.length,
          renderCount,
          applyModeCount,
          first,
          second,
          third,
          firstFresh,
          secondFresh,
          thirdFresh,
          cacheSize: fileCandidateCache.size,
          sources: currentFileCandidateEntries().map((entry) => entry.source || ""),
          literalFiles,
          literalRel,
          literalSuffix,
          literalCandidates,
          literalCacheKey,
        }));
      })().catch((err) => { console.error(err && err.stack || err); process.exit(1); });
    '''
    return run_file_candidate_refresh_probe(setup_js, body_js)


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

    def test_tokenized_search_result_suppresses_display_only_recent_duplicate(self) -> None:
        result = eval_file_picker_search_helpers(
            {
                "fileEntries": [
                    {"path": r"bad\xffname.txt", "changed": False, "source": "recent", "gitPath": False},
                ],
                "filePickerSearchActive": True,
                "filePickerInputValue": "bad",
                "fileSearchLoadedQuery": "bad",
                "fileSearchResults": [{"path": r"bad\xffname.txt", "apiPath": "codoxear-git-path-bytes-v1:YmFk_25hbWUudHh0", "score": 12000}],
            }
        )
        file_entries = [entry for entry in result["entries"] if not entry.get("createNew")]
        self.assertEqual(
            [(entry["path"], entry["gitPath"], entry["apiPath"]) for entry in file_entries],
            [(r"bad\xffname.txt", False, "codoxear-git-path-bytes-v1:YmFk_25hbWUudHh0")],
        )

    def test_search_state_request_keeps_literal_and_tokenized_same_display_path(self) -> None:
        # createSearchState.request dedupes API matches by full identity
        # (apiPath-or-path), not display path. A literal ``bad\xffname.txt``
        # (no token) and a raw-byte tokenized ``bad<ff>name.txt`` share the same
        # JSON-safe display string but are distinct files; both must survive.
        display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
        file_helpers_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
        file_picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
        display_path = r"bad\xffname.txt"
        token = "codoxear-git-path-bytes-v1:YmFk_25hbWUudHh0"
        matches_json = json.dumps(
            [
                {"path": display_path, "api_path": "", "score": 100},
                {"path": display_path, "api_path": token, "score": 12000},
            ]
        )
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{
              window: {{}},
            }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(display_source)}, ctx);
            vm.runInContext({json.dumps(file_helpers_source)}, ctx);
            vm.runInContext({json.dumps(file_picker_source)}, ctx);
            const matches = {matches_json};
            const host = {{
              blocked: () => false,
              currentSessionId: () => "s",
              api: async () => ({{ matches, truncated: false }}),
              inputValue: () => "bad",
              isMenuOpen: () => true,
              renderMenu: () => {{}},
              applyMenuState: () => {{}},
              normalizeFileApiPath: (value) => typeof value === "string" && value !== "" ? value : "",
              setTimeout: () => 0,
              clearTimeout: () => {{}},
            }};
            const state = ctx.window.CodoxearFilePicker.createSearchState(host);
            state.setSessionId("s");
            void state.request("bad").then(() => {{
              const snap = state.snapshot();
              process.stdout.write(JSON.stringify({{ results: snap.results }}));
            }});
            """
        )
        proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = json.loads(proc.stdout)
        identities = sorted((entry["path"], entry["apiPath"]) for entry in result["results"])
        self.assertEqual(
            identities,
            sorted([
                (display_path, ""),
                (display_path, token),
            ]),
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
        picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
        self.assertNotIn("function localFilePickerSearchEntries(query)", source)
        self.assertNotIn("function prependDraftFileEntry(entries, query)", source)
        self.assertIn("function localFilePickerSearchEntries(context, query)", picker_source)
        self.assertIn("function prependDraftFileEntry(entries, query, context)", picker_source)
        self.assertNotIn('appendFilePickerStatusRow("Searching full project...");', source)
        self.assertIn('appendStatus("Searching full project...");', picker_source)
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
        self.assertEqual(result["gitStateMessage"], "Not a git repository \u2014 no changed files")
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
        self.assertEqual(result["callsAfterKnown"], ["/api/sessions/s1/git/changed_files"])
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
        picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
        css = (APP_JS.parent / "app.css").read_text(encoding="utf-8")
        start = picker_source.index("function appendHighlightedFileMenuPath(parent, text, query, host = {}) {")
        end = picker_source.index("function appendFilePickerSection", start)
        block = picker_source[start:end]
        self.assertIn("span.appendChild(createTextNode(value.slice(cursor, start)));", block)
        self.assertIn('span.appendChild(createEl("mark", { class: "fileMenuMatch", text: value.slice(start, end) }));', block)
        self.assertIn("parent.appendChild(span);", block)
        self.assertNotIn("innerHTML", block)
        self.assertNotIn("function appendHighlightedFileMenuPath(parent, text, query) {", source)
        self.assertNotIn("document.createTextNode(value.slice(cursor", source)
        self.assertIn("appendHighlightedFileMenuPath(btn, path, query, host);", picker_source)
        self.assertIn("appendHighlightedFileMenuPath(btn, `Create new file: ${path}`, query, host);", picker_source)
        self.assertIn("titleForEntry: (entry, hint) => filePickerTitle(entry, hint)", source)
        self.assertIn("title,", picker_source)
        self.assertIn(".fileMenuMatch", css)

    def test_file_picker_candidate_sections_and_cache_are_present(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
        picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
        viewer_source = APP_FILE_VIEWER_JS.read_text(encoding="utf-8")
        css = (APP_JS.parent / "app.css").read_text(encoding="utf-8")
        self.assertIn("function filePickerSectionLabel(source)", source)
        self.assertIn("return codoxearFileHelpers.filePickerSectionLabel(source);", source)
        self.assertIn('return "Changed files";', helper_source)
        self.assertIn('return "Mentioned in chat";', helper_source)
        self.assertIn('return "Recently opened";', helper_source)
        self.assertNotIn('return "Changed files";', source)
        self.assertIn("let fileCandidateCache = new Map();", viewer_source)
        self.assertNotIn("const fileCandidateCache = new Map();", source)
        self.assertIn("const FILE_CANDIDATE_CACHE_TTL_MS = 15000;", source)
        self.assertIn("fileCandidateCache.set(sid, { key, ts: Number(now || 0), entries: currentFileCandidateEntries() });", viewer_source)
        self.assertIn("const openedFileRuntime = codoxearFileViewer.createOpenedFileRuntime({", source)
        self.assertIn("deleteCandidateCache: (sessionId) => fileViewerController.deleteFileCandidateCache(sessionId)", source)
        self.assertIn("deleteCandidateCache(sid);", viewer_source)
        self.assertIn("openFilePathWithResolvedMode(selectedEntry.path, { line: filePickerSelectionLine(), changed: Boolean(selectedEntry.changed), gitPath: Boolean(selectedEntry.gitPath), apiPath: selectedEntry.apiPath })", source)
        self.assertIn("void openFilePathWithResolvedMode(active.path, { line: selectionLine(input.value), changed: Boolean(active.changed), gitPath: Boolean(active.gitPath), apiPath: active.apiPath })", picker_source)
        self.assertIn("const filePickerInputRuntime = codoxearFilePicker.createInputRuntime({", source)
        self.assertIn("compareFilePickerEntries", source)
        self.assertIn("filePickerMenuState = codoxearFilePicker.createMenuState", source)
        self.assertIn("filePickerSearchState = codoxearFilePicker.createSearchState", source)
        self.assertIn("function createMenuState(host = {})", picker_source)
        self.assertIn("function createSearchState(host)", picker_source)
        self.assertIn("function visibleFilePickerEntries(context)", picker_source)
        self.assertNotIn("let fileMenuOpen", source)
        self.assertNotIn("let fileMenuFocus", source)
        self.assertNotIn("let filePickerSearchActive", source)
        self.assertNotIn("let filePickerReferenceLineQuery", source)
        self.assertNotIn("let filePickerReferenceLine", source)
        self.assertNotIn("let filePickerPreserveSearchOnFocus", source)
        self.assertNotIn("let filePickerSuppressDraftQuery", source)
        self.assertIn("prependPendingSessionPathEntry(localFilePickerSearchEntries(context, query), query)", picker_source)
        self.assertIn("function filePickerCandidateScore(path, query)", source)
        self.assertIn("function applyFreshFileCandidateCache", viewer_source)
        self.assertIn("return applyFileCandidateRefreshEntries(cached.entries, { gitStateFresh: false });", viewer_source)
        self.assertIn("const fileCandidateRefreshRuntime = codoxearFileViewer.createFileCandidateRefreshRuntime({", source)
        self.assertIn("return await fileCandidateRefreshRuntime.refresh({ force, sessionId, syncToken });", source)
        self.assertNotIn("fileViewerController.applyFreshFileCandidateCache(sid, cacheKey", source)
        self.assertIn("applyFreshCache(sid, cacheKey", viewer_source)
        self.assertNotIn("fileViewerController.applyFileCandidateRefreshEntries(merged, { gitStateFresh: changedEntriesFresh });", source)
        self.assertIn("applyRefreshEntries(merged, { gitStateFresh: changedEntriesFresh, gitStateMessage });", viewer_source)
        self.assertNotIn("const diffable = canToggleMode && activeFileGitPathValue() && fileCandidateGitStateFresh", source)
        self.assertIn("function currentFileModeControlState", viewer_source)
        self.assertIn("const diffable = Boolean(canToggleMode && identity.gitPath && currentFileCandidateGitStateFresh()", viewer_source)
        self.assertIn("const canUseDiffView = request && request.gitPath && currentFileCandidateGitStateFresh()", viewer_source)
        self.assertNotIn("function fileCandidateKeyForEntry(entry)", source)
        self.assertIn("fileCandidateKeyForEntry(entry)", viewer_source)
        self.assertIn("fileEntryMap.has(next.key)", viewer_source)
        self.assertIn(".fileMenuSection", css)
        self.assertIn("function filePickerIdentityHint(entry, duplicatePaths, options)", source)
        self.assertIn("function duplicateFilePickerPaths(entries)", source)
        self.assertNotIn('"aria-label": filePickerTitle(entry, identityHint)', source)
        self.assertIn("titleForEntry: (entry, hint) => filePickerTitle(entry, hint)", source)
        self.assertIn("fileMenuHint fileMenuIdentity", picker_source)
        self.assertIn(".fileMenuIdentity", css)


if __name__ == "__main__":
    unittest.main()
