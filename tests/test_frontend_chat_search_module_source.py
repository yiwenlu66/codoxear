import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_CHAT_SEARCH_JS = ROOT / "codoxear" / "static" / "app_chat_search.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def run_node_json(js: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "TZ": "UTC"},
    )
    return json.loads(proc.stdout)


HARNESS = r"""
const vm = require("vm");

// --- Fake timers so the 300ms all-count debounce fires deterministically. ---
const timers = [];
let nextTimerId = 1;
function fakeSetTimeout(fn, delay) {
  const id = nextTimerId++;
  timers.push({ id, fn });
  return id;
}
function fakeClearTimeout(id) {
  const idx = timers.findIndex((t) => t.id === id);
  if (idx >= 0) timers.splice(idx, 1);
}
function flushTimers() {
  while (timers.length) {
    const t = timers.shift();
    t.fn();
  }
}

// --- State recorded by fakes, resettable per-test. ---
const calls = [];
const toasts = [];
const focusCalls = [];
const selectCalls = [];
const scrollCalls = [];
const clearMarksCalls = [];
const applyMarksCalls = [];
const pulseCalls = [];
const syncTimeCalls = [];
const invalidateCalls = [];
const setOlderStateCalls = [];
const showOlderErrorCalls = [];
const openSessionCalls = [];
const authLossCalls = [];
const renderDetachedCalls = [];
const olderLoadFinishes = [];
const apiCalls = [];

let selected = null;
let pollGen = 0;
let reducedMotion = false;
let olderMessages = true;
let loadingOlder = false;
let olderLimit = 50;
let oldestCursor = "cursor-old";
let apiHandler = null;
let loadOlderImpl = null;

function makeNode(id) {
  return {
    _id: id,
    style: { display: "" },
    disabled: false,
    textContent: "",
    title: "",
    value: "",
    onclick: null,
    oninput: null,
    onkeydown: null,
    focus(opts) { focusCalls.push({ id, opts }); },
    select() { selectCalls.push(id); },
    dataset: {},
  };
}

const dom = {
  chatSearchBtn: makeNode("chatSearchBtn"),
  chatSearchInput: makeNode("chatSearchInput"),
  chatSearchPrevBtn: makeNode("chatSearchPrevBtn"),
  chatSearchNextBtn: makeNode("chatSearchNextBtn"),
  chatSearchCloseBtn: makeNode("chatSearchCloseBtn"),
  chatSearchStatus: makeNode("chatSearchStatus"),
  chatSearchAllHintEl: makeNode("chatSearchAllHintEl"),
  chatSearchBar: makeNode("chatSearchBar"),
};

let currentRows = [];
function makeRow(name, text, opts = {}) {
  return {
    name,
    text,
    order: typeof opts.order === "number" ? opts.order : 0,
    dataset: {
      historyCursor: opts.historyCursor || "",
      searchForcedQuery: opts.searchForcedQuery || "",
    },
    scrollIntoView(o) { scrollCalls.push({ name, opts: o }); },
  };
}

let loadId = 0;
const olderLoadRuntime = {
  beginLoad(opts) {
    const load = { id: ++loadId, signal: { aborted: false, addEventListener() {} }, opts };
    return load;
  },
  isCurrent(load) { return Boolean(load); },
  finishLoad(load) { olderLoadFinishes.push(load); },
};

function api(url, options) {
  apiCalls.push({ url, options });
  if (apiHandler) return apiHandler(url, options);
  return Promise.resolve({});
}

async function loadOlderMessages(opts) {
  if (loadOlderImpl) return await loadOlderImpl(opts);
  return false;
}

const deps = {
  chatSearchBtn: dom.chatSearchBtn,
  chatSearchInput: dom.chatSearchInput,
  chatSearchPrevBtn: dom.chatSearchPrevBtn,
  chatSearchNextBtn: dom.chatSearchNextBtn,
  chatSearchCloseBtn: dom.chatSearchCloseBtn,
  chatSearchStatus: dom.chatSearchStatus,
  chatSearchAllHintEl: dom.chatSearchAllHintEl,
  chatSearchBar: dom.chatSearchBar,
  // Real transcript search runtimes from app_transcript.js.
  createLoadedChatSearchRuntime: null, // filled after transcript module load
  createChatSearchAllRuntime: null,
  getSelected: () => selected,
  getPollGen: () => pollGen,
  api,
  setToast: (t) => { toasts.push(t); calls.push(["setToast", t]); },
  openSession: async (sid, opts) => { openSessionCalls.push({ sid, opts }); },
  handleAppAuthLoss: () => { authLossCalls.push(1); },
  chatSearchTranscriptHint: (match, query) => (match ? `hint:${match.text}:${query}` : ""),
  syncVisibleTimeIndicator: () => { syncTimeCalls.push(1); },
  renderedMessageRows: () => currentRows.slice(),
  rowSearchText: (row) => (row ? row.text : ""),
  compareRowsInDomOrder: (a, b) => (a ? a.order : 0) - (b ? b.order : 0),
  clearChatSearchMarks: () => { clearMarksCalls.push(1); },
  applyChatSearchMarks: (matches, currentRow) => { applyMarksCalls.push({ matches: matches ? matches.length : 0, current: currentRow ? currentRow.name : null }); },
  pulseNavigatedRow: (row) => { pulseCalls.push(row ? row.name : null); },
  prefersReducedMotion: () => reducedMotion,
  oldestRenderedHistoryCursor: () => oldestCursor,
  renderDetachedTranscriptWindow: (events, opts) => { renderDetachedCalls.push({ events: events.length, hasMore: opts && opts.hasMore }); return true; },
  invalidateOlderLoad: () => { invalidateCalls.push(1); },
  setOlderState: (state) => { setOlderStateCalls.push(state); },
  showOlderLoadError: () => { showOlderErrorCalls.push(1); },
  hasOlderMessages: () => olderMessages,
  isLoadingOlderMessages: () => loadingOlder,
  olderPageLimit: () => olderLimit,
  loadOlderMessages,
  olderLoadRuntime,
};

const ctx = {
  HTMLElement: function HTMLElement() {},
  AbortController,
  document: {},
  console,
  window: { setTimeout: fakeSetTimeout, clearTimeout: fakeClearTimeout },
};
vm.createContext(ctx);

// Load real transcript helpers so the search runtimes are the production
// implementations (createLoadedChatSearchRuntime / createChatSearchAllRuntime).
vm.runInContext(TRANSCRIPT_SOURCE, ctx);
deps.createLoadedChatSearchRuntime = ctx.window.CodoxearTranscript.createLoadedChatSearchRuntime;
deps.createChatSearchAllRuntime = ctx.window.CodoxearTranscript.createChatSearchAllRuntime;

// Load the chat search controller module under test.
vm.runInContext(MODULE_SOURCE, ctx);
const controller = ctx.window.CodoxearChatSearch.createChatSearchController(deps);

globalThis.__harness = {
  controller,
  dom,
  calls,
  toasts,
  focusCalls,
  selectCalls,
  scrollCalls,
  clearMarksCalls,
  applyMarksCalls,
  pulseCalls,
  syncTimeCalls,
  invalidateCalls,
  setOlderStateCalls,
  showOlderErrorCalls,
  openSessionCalls,
  authLossCalls,
  renderDetachedCalls,
  olderLoadFinishes,
  apiCalls,
  flushTimers,
  select: (sid) => { selected = sid; },
  selected: () => selected,
  setPollGen: (g) => { pollGen = g; },
  setReducedMotion: (v) => { reducedMotion = v; },
  setOlderMessages: (v) => { olderMessages = v; },
  setLoadingOlder: (v) => { loadingOlder = v; },
  setOlderLimit: (v) => { olderLimit = v; },
  setOldestCursor: (v) => { oldestCursor = v; },
  setRows: (rows) => { currentRows = rows; },
  setInput: (v) => { dom.chatSearchInput.value = v; },
  seedQuery: (q) => { dom.chatSearchInput.value = q; currentRows = []; controller.refreshLoaded({ jump: false, preserveCurrent: false }); },
  setApiHandler: (fn) => { apiHandler = fn; },
  setLoadOlderImpl: (fn) => { loadOlderImpl = fn; },
  makeRow,
  reset: () => {
    calls.length = 0; toasts.length = 0; focusCalls.length = 0; selectCalls.length = 0;
    scrollCalls.length = 0; clearMarksCalls.length = 0; applyMarksCalls.length = 0;
    pulseCalls.length = 0; syncTimeCalls.length = 0; invalidateCalls.length = 0;
    setOlderStateCalls.length = 0; showOlderErrorCalls.length = 0; openSessionCalls.length = 0;
    authLossCalls.length = 0; renderDetachedCalls.length = 0; olderLoadFinishes.length = 0;
    apiCalls.length = 0; apiHandler = null; loadOlderImpl = null;
    dom.chatSearchStatus.textContent = ""; dom.chatSearchAllHintEl.textContent = "";
    dom.chatSearchAllHintEl.title = ""; dom.chatSearchAllHintEl.style.display = "";
    dom.chatSearchPrevBtn.disabled = false; dom.chatSearchNextBtn.disabled = false;
  },
};
"""


def harness_script(epilogue: str) -> str:
    module_source = APP_CHAT_SEARCH_JS.read_text(encoding="utf-8")
    transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
    js = (
        textwrap.dedent(
            f"""
        const MODULE_SOURCE = {json.dumps(module_source)};
        const TRANSCRIPT_SOURCE = {json.dumps(transcript_source)};
        """
        )
        + HARNESS
        + "\n(async () => {\n"
        + textwrap.dedent(epilogue)
        + "\n})().then(() => {\n"
        + "  process.stdout.write(JSON.stringify(globalThis.__result || {}));\n"
        + "}).catch((err) => {\n"
        + "  console.error(err && err.stack || err);\n"
        + "  process.exit(1);\n"
        + "});\n"
    )
    return js


class TestFrontendChatSearchModuleSource(unittest.TestCase):
    # --- 1. frozen export + missing dep failures ---

    def test_module_export_is_frozen_createChatSearch_controller(self) -> None:
        module_source = APP_CHAT_SEARCH_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}}, AbortController, document: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(module_source)}, ctx);
            process.stdout.write(JSON.stringify({{
              frozen: Object.isFrozen(ctx.window.CodoxearChatSearch),
              keys: Object.keys(ctx.window.CodoxearChatSearch),
              hasCreate: typeof ctx.window.CodoxearChatSearch.createChatSearchController === "function",
            }}));
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["frozen"])
        self.assertEqual(result["keys"], ["createChatSearchController"])
        self.assertTrue(result["hasCreate"])

    def test_create_throws_on_missing_deps(self) -> None:
        module_source = APP_CHAT_SEARCH_JS.read_text(encoding="utf-8")
        js = textwrap.dedent(
            f"""
            const vm = require("vm");
            const ctx = {{ window: {{}}, HTMLElement: function HTMLElement() {{}}, AbortController, document: {{}} }};
            vm.createContext(ctx);
            vm.runInContext({json.dumps(module_source)}, ctx);
            const create = ctx.window.CodoxearChatSearch.createChatSearchController;
            function node(id) {{ return {{ style: {{}}, dataset: {{}}, focus() {{}}, select() {{}} }}; }}
            const good = {{
              chatSearchBtn: node("btn"),
              chatSearchInput: node("input"),
              chatSearchPrevBtn: node("prev"),
              chatSearchNextBtn: node("next"),
              chatSearchCloseBtn: node("close"),
              chatSearchStatus: node("status"),
              chatSearchAllHintEl: node("hint"),
              chatSearchBar: node("bar"),
              createLoadedChatSearchRuntime: () => ({{}}),
              createChatSearchAllRuntime: () => ({{}}),
              getSelected: () => null,
              getPollGen: () => 0,
              api: () => ({{}}),
              setToast: () => {{}},
              openSession: () => {{}},
              handleAppAuthLoss: () => {{}},
              chatSearchTranscriptHint: () => "",
              syncVisibleTimeIndicator: () => {{}},
              renderedMessageRows: () => [],
              rowSearchText: () => "",
              compareRowsInDomOrder: () => 0,
              clearChatSearchMarks: () => {{}},
              applyChatSearchMarks: () => {{}},
              pulseNavigatedRow: () => {{}},
              prefersReducedMotion: () => false,
              oldestRenderedHistoryCursor: () => "",
              renderDetachedTranscriptWindow: () => true,
              invalidateOlderLoad: () => {{}},
              setOlderState: () => {{}},
              showOlderLoadError: () => {{}},
              hasOlderMessages: () => false,
              isLoadingOlderMessages: () => false,
              olderPageLimit: () => 50,
              loadOlderMessages: () => false,
              olderLoadRuntime: {{ beginLoad() {{}}, isCurrent() {{}}, finishLoad() {{}} }},
            }};
            const errors = [];
            for (const key of Object.keys(good)) {{
              const partial = {{ ...good }};
              delete partial[key];
              try {{ create(partial); errors.push(null); }}
              catch (err) {{ errors.push(err && err.name === "TypeError" ? key : (err && err.name)); }}
            }}
            try {{ create(); errors.push("options-ok"); }} catch (err) {{ errors.push(err && err.name === "TypeError" ? "options" : (err && err.name)); }}
            process.stdout.write(JSON.stringify({{ errors }}));
            """
        )
        result = run_node_json(js)
        self.assertNotIn(None, result["errors"])
        for required in [
            "chatSearchInput", "createLoadedChatSearchRuntime", "createChatSearchAllRuntime",
            "getSelected", "api", "setToast", "openSession", "handleAppAuthLoss",
            "chatSearchTranscriptHint", "syncVisibleTimeIndicator", "renderedMessageRows",
            "rowSearchText", "compareRowsInDomOrder", "clearChatSearchMarks", "applyChatSearchMarks",
            "pulseNavigatedRow", "prefersReducedMotion", "oldestRenderedHistoryCursor",
            "renderDetachedTranscriptWindow", "invalidateOlderLoad", "setOlderState",
            "showOlderLoadError", "hasOlderMessages", "isLoadingOlderMessages", "olderPageLimit",
            "loadOlderMessages", "olderLoadRuntime", "options",
        ]:
            self.assertIn(required, result["errors"])

    # --- 2. open/close/focus/select/display/sync-time ---

    def test_open_noop_when_no_selected(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.controller.open();
            globalThis.__result = { display: h.dom.chatSearchBar.style.display, syncTime: h.syncTimeCalls.length, focus: h.focusCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertNotEqual(result["display"], "flex")
        self.assertEqual(result["syncTime"], 0)
        self.assertEqual(result["focus"], 0)

    def test_open_with_selected_displays_bar_syncs_time_focuses_and_selects(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.open();
            globalThis.__result = {
              open: h.controller.isOpen(),
              display: h.dom.chatSearchBar.style.display,
              syncTime: h.syncTimeCalls.length,
              focus: h.focusCalls,
              select: h.selectCalls,
            };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["open"])
        self.assertEqual(result["display"], "flex")
        self.assertEqual(result["syncTime"], 1)
        self.assertEqual(result["focus"], [{"id": "chatSearchInput", "opts": {"preventScroll": True}}])
        self.assertEqual(result["select"], ["chatSearchInput"])

    def test_close_hides_bar_clears_marks_resets_all_and_syncs_time(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.controller.open();
            h.reset();
            h.controller.close();
            globalThis.__result = {
              open: h.controller.isOpen(),
              display: h.dom.chatSearchBar.style.display,
              clearMarks: h.clearMarksCalls.length,
              syncTime: h.syncTimeCalls.length,
            };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["open"])
        self.assertEqual(result["display"], "none")
        self.assertEqual(result["clearMarks"], 1)
        self.assertEqual(result["syncTime"], 1)

    # --- 3. empty query clears marks/matches/all-count/status ---

    def test_empty_query_clears_matches_and_status_loaded(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("");
            h.setRows([h.makeRow("r1", "needle text")]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: true });
            globalThis.__result = {
              status: h.dom.chatSearchStatus.textContent,
              matches: h.controller.currentMatches().length,
              clearMarks: h.clearMarksCalls.length,
              query: h.controller.currentQuery(),
              prevDisabled: h.dom.chatSearchPrevBtn.disabled,
              nextDisabled: h.dom.chatSearchNextBtn.disabled,
            };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["status"], "Loaded")
        self.assertEqual(result["matches"], 0)
        self.assertTrue(result["clearMarks"] >= 1)
        self.assertEqual(result["query"], "")
        self.assertTrue(result["prevDisabled"])
        self.assertTrue(result["nextDisabled"])

    # --- 4. loaded match refresh from row text + forced-query + mark ---

    def test_refresh_loaded_matches_by_row_text_and_applies_marks(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([
              h.makeRow("r1", "has needle here", { order: 1 }),
              h.makeRow("r2", "no match", { order: 2 }),
              h.makeRow("r3", "NEEDLE uppercase", { order: 3 }),
            ]);
            h.controller.refreshLoaded({ jump: true, preserveCurrent: false });
            globalThis.__result = {
              status: h.dom.chatSearchStatus.textContent,
              matches: h.controller.currentMatches().length,
              applyMarks: h.applyMarksCalls,
              pulses: h.pulseCalls,
            };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["status"], "1/2 loaded")
        self.assertEqual(result["matches"], 2)
        self.assertEqual(len(result["applyMarks"]), 1)
        self.assertEqual(result["applyMarks"][0]["matches"], 2)
        self.assertEqual(len(result["pulses"]), 1)

    def test_refresh_loaded_matches_forced_query_row(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("forced");
            h.setRows([
              h.makeRow("r1", "unrelated text", { order: 1, searchForcedQuery: "forced" }),
              h.makeRow("r2", "no match", { order: 2 }),
            ]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            globalThis.__result = {
              matches: h.controller.currentMatches().length,
              status: h.dom.chatSearchStatus.textContent,
            };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["status"], "1/1 loaded")

    # --- 5. status / all-hint projection ---

    def test_status_loading_older_suffix(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            // Drive loading-older suffix by invoking the cursor-window loader.
            h.setApiHandler(() => Promise.resolve({ events: [{}, {}], has_older: false }));
            await h.controller.loadChatSearchCursorWindow("c1", { targetHistoryCursor: "" });
            globalThis.__result = { status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        # After the cursor window settles, loadingOlder is false again; the
        # status reflects the loaded match count with no loading suffix.
        self.assertIn("loaded", result["status"])

    def test_status_all_count_suffix_and_hint(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.setApiHandler((url) => {
              if (url.indexOf("count_max=1000") >= 0) {
                return Promise.resolve({ matches: [{ text: "needle found", role: "user" }], match_count: 5, match_count_truncated: false });
              }
              return Promise.resolve({});
            });
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = {
              status: h.dom.chatSearchStatus.textContent,
              hint: h.dom.chatSearchAllHintEl.textContent,
              hintDisplay: h.dom.chatSearchAllHintEl.style.display,
              nextDisabled: h.dom.chatSearchNextBtn.disabled,
            };
            """
        )
        result = run_node_json(js)
        self.assertIn("· 5 all", result["status"])
        self.assertEqual(result["hint"], "all: hint:needle found:needle")
        self.assertNotEqual(result["hintDisplay"], "none")
        # Next button enabled because loaded matches exist.
        self.assertFalse(result["nextDisabled"])

    def test_status_all_count_truncated_suffix(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.setApiHandler(() => Promise.resolve({ matches: [], match_count: 1000, match_count_truncated: true }));
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        self.assertIn("· 1000+ all", result["status"])

    def test_next_button_enabled_when_only_older_matches_available(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]); // no loaded matches
            h.setApiHandler(() => Promise.resolve({ matches: [], match_count: 3, match_count_truncated: false }));
            h.setOlderMessages(true);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { prevDisabled: h.dom.chatSearchPrevBtn.disabled, nextDisabled: h.dom.chatSearchNextBtn.disabled };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["prevDisabled"])
        self.assertFalse(result["nextDisabled"])

    # --- 6. all-count schedule/reset/fetch stale guards, AbortError, failRequest ---

    def test_all_count_request_url_includes_count_max(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.setApiHandler(() => Promise.resolve({ matches: [], match_count: 0, match_count_truncated: false }));
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            const searchCall = h.apiCalls.find((c) => c.url.indexOf("count_max=1000") >= 0);
            globalThis.__result = { url: searchCall ? searchCall.url : null };
            """
        )
        result = run_node_json(js)
        self.assertIn("limit=1&text_max=96&count_max=1000", result["url"])
        self.assertIn("q=needle", result["url"])

    def test_all_count_stale_selected_does_not_complete(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.setApiHandler((url) => {
              // Flip selected mid-flight to invalidate the stale guard.
              h.select("sid-other");
              return Promise.resolve({ matches: [], match_count: 9, match_count_truncated: false });
            });
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        # Stale response is dropped: no "9 all" suffix.
        self.assertNotIn("9 all", result["status"])

    def test_all_count_abort_error_returns_silently(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            const abortErr = new Error("aborted");
            abortErr.name = "AbortError";
            h.setApiHandler(() => Promise.reject(abortErr));
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { ok: true, status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        # AbortError does not call failRequest; count stays null so no suffix.
        self.assertNotIn("all", result["status"])

    def test_all_count_non_stale_error_calls_fail_request(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.setApiHandler(() => Promise.reject(new Error("network")));
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { allSnapshot: h.controller.allSnapshot(), status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        # After a non-AbortError, the all runtime resets count (failRequest sets count null).
        self.assertIsNone(result["allSnapshot"]["count"])

    # --- 7. step no-query / no-loaded / no-older toasts + loaded next/prev ---

    def test_step_no_query_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("");
            await h.controller.step(1);
            globalThis.__result = { toasts: h.toasts.slice() };
            """
        )
        result = run_node_json(js)
        self.assertIn("Enter a loaded-chat search", result["toasts"])

    def test_step_query_no_loaded_no_older_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.setOlderMessages(false);
            await h.controller.step(1);
            globalThis.__result = { toasts: h.toasts.slice() };
            """
        )
        result = run_node_json(js)
        self.assertIn("No loaded matches", result["toasts"])

    def test_step_query_no_loaded_older_unavailable_after_load_toast(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.setOlderMessages(true);
            // all-count reports 3 matches exist; nearest-window + loadOlder fail.
            h.setApiHandler(() => Promise.resolve({ matches: [], match_count: 0, match_count_truncated: false }));
            h.setLoadOlderImpl(() => { h.setOlderMessages(false); return false; });
            // Pre-seed the all-count to a positive value so step attempts older loading.
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.flushTimers();
            await new Promise((r) => setTimeout(r, 0));
            h.setApiHandler((url) => {
                if (url.indexOf("order=latest") >= 0) return Promise.resolve({ matches: [] });
                return Promise.resolve({});
            });
            h.reset();
            await h.controller.step(1);
            globalThis.__result = { toasts: h.toasts.slice() };
            """
        )
        result = run_node_json(js)
        # The all-count for this query is 0 so step goes straight to "No loaded matches".
        self.assertIn(result["toasts"][0], ["No loaded matches", "No loaded matches after loading older messages"])

    def test_step_loaded_next_focuses_next_match(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([
              h.makeRow("r1", "needle one", { order: 1 }),
              h.makeRow("r2", "needle two", { order: 2 }),
              h.makeRow("r3", "needle three", { order: 3 }),
            ]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.reset();
            await h.controller.step(1);
            globalThis.__result = { pulses: h.pulseCalls, scrolls: h.scrollCalls, applyMarksAtLeast: h.applyMarksCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["pulses"], ["r2"])
        self.assertGreaterEqual(result["applyMarksAtLeast"], 1)

    def test_step_loaded_prev_at_first_wraps_to_last(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([
              h.makeRow("r1", "needle one", { order: 1 }),
              h.makeRow("r2", "needle two", { order: 2 }),
            ]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.reset();
            await h.controller.step(-1); // at index 0 -> wraps to last
            globalThis.__result = { pulses: h.pulseCalls };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["pulses"], ["r2"])

    # --- 8 + 9. loadOlderUntilChatSearchMatch page loop / boundary / finally ---

    def test_load_older_until_finds_match_and_clears_loading_older(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            let pages = 0;
            h.setLoadOlderImpl(() => {
              pages += 1;
              if (pages === 1) h.setRows([h.makeRow("r1", "needle found", { order: 1 })]);
              h.setOlderMessages(false);
              return true;
            });
            const found = await h.controller.loadOlderUntilChatSearchMatch();
            globalThis.__result = { found, pulses: h.pulseCalls, status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["found"])
        self.assertEqual(result["pulses"], ["r1"])
        # loadingOlder cleared in finally -> no "loading older" suffix.
        self.assertNotIn("loading older", result["status"])

    def test_load_older_until_boundary_focus_last(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            // Seed a boundary match (a row object) that the loop will receive.
            const boundary = h.makeRow("boundary", "needle boundary", { order: 5 });
            let pages = 0;
            h.setLoadOlderImpl(() => {
              pages += 1;
              if (pages === 1) h.setRows([
                h.makeRow("r1", "needle one", { order: 1 }),
                boundary,
                h.makeRow("r2", "needle two", { order: 2 }),
              ]);
              h.setOlderMessages(false);
              return true;
            });
            const found = await h.controller.loadOlderUntilChatSearchMatch({ boundaryMatch: boundary, focus: "last" });
            // boundaryIndex > 0 -> focus = last => boundaryIndex-1 = r1 (index 0).
            globalThis.__result = { found, pulses: h.pulseCalls };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["found"])
        self.assertEqual(result["pulses"], ["r1"])

    def test_load_older_until_returns_false_when_no_older(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setOlderMessages(false);
            const found = await h.controller.loadOlderUntilChatSearchMatch();
            globalThis.__result = { found, status: h.dom.chatSearchStatus.textContent };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["found"])
        self.assertNotIn("loading older", result["status"])

    # --- 10. loadNearestOlderChatSearchWindow API URL / 401 / 409 / stale / cursor ---

    def test_load_nearest_older_window_search_url_and_delegates_to_cursor_window(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setOldestCursor("CUR");
            h.setOlderMessages(true);
            h.setApiHandler((url) => {
              if (url.indexOf("order=latest") >= 0) {
                return Promise.resolve({ matches: [{ load_cursor: "LDC", history_cursor: "HIST" }] });
              }
              if (url.indexOf("/messages/history") >= 0) {
                h.setRows([h.makeRow("target", "needle target", { order: 1, historyCursor: "HIST" })]);
                return Promise.resolve({ events: [{}, {}], has_older: false });
              }
              return Promise.resolve({});
            });
            const ok = await h.controller.loadNearestOlderChatSearchWindow();
            const searchCall = h.apiCalls.find((c) => c.url.indexOf("order=latest") >= 0);
            const historyCall = h.apiCalls.find((c) => c.url.indexOf("/messages/history") >= 0);
            globalThis.__result = { ok, searchUrl: searchCall.url, historyUrl: historyCall ? historyCall.url : null, detached: h.renderDetachedCalls };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        self.assertIn("order=latest&before=CUR", result["searchUrl"])
        self.assertIn("q=needle", result["searchUrl"])
        self.assertIn("cursor=LDC", result["historyUrl"])
        self.assertEqual(len(result["detached"]), 1)

    def test_load_nearest_older_window_401_triggers_auth_loss(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setOldestCursor("CUR");
            const err = new Error("unauth"); err.status = 401;
            h.setApiHandler(() => Promise.reject(err));
            const ok = await h.controller.loadNearestOlderChatSearchWindow();
            globalThis.__result = { ok, authLoss: h.authLossCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["authLoss"], 1)

    def test_load_nearest_older_window_409_refreshes_session(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setOldestCursor("CUR");
            const err = new Error("rebind"); err.status = 409;
            h.setApiHandler(() => Promise.reject(err));
            const ok = await h.controller.loadNearestOlderChatSearchWindow();
            globalThis.__result = { ok, openSession: h.openSessionCalls };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["openSession"], [{"sid": "sid-1", "opts": {"useCache": False}}])

    def test_load_nearest_older_window_stale_after_fetch_returns_false(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setOldestCursor("CUR");
            h.setApiHandler((url) => {
              if (url.indexOf("order=latest") >= 0) {
                h.select("sid-other"); // invalidate stale guard
                return Promise.resolve({ matches: [{ load_cursor: "LDC", history_cursor: "HIST" }] });
              }
              return Promise.resolve({});
            });
            const ok = await h.controller.loadNearestOlderChatSearchWindow();
            globalThis.__result = { ok };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])

    def test_load_nearest_older_window_no_cursor_returns_false(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setOldestCursor("");
            const ok = await h.controller.loadNearestOlderChatSearchWindow();
            globalThis.__result = { ok, apiCalls: h.apiCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["apiCalls"], 0)

    # --- 11. loadChatSearchCursorWindow history API / render / target / toast / errors / finally ---

    def test_load_cursor_window_renders_and_toasts_loaded_match(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            // Target row present among rendered rows after detach render.
            const target = h.makeRow("target", "needle target", { order: 1, historyCursor: "HIST" });
            h.setApiHandler((url) => {
              if (url.indexOf("/messages/history") >= 0) {
                h.setRows([target]);
                return Promise.resolve({ events: [{}, {}], has_older: true });
              }
              return Promise.resolve({});
            });
            const ok = await h.controller.loadChatSearchCursorWindow("LDC", { targetHistoryCursor: "HIST" });
            const historyCall = h.apiCalls.find((c) => c.url.indexOf("/messages/history") >= 0);
            globalThis.__result = { ok, historyUrl: historyCall.url, detached: h.renderDetachedCalls, toasts: h.toasts, pulses: h.pulseCalls, invalidate: h.invalidateCalls.length, finishes: h.olderLoadFinishes.length };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["ok"])
        self.assertIn("cursor=LDC", result["historyUrl"])
        self.assertEqual(len(result["detached"]), 1)
        self.assertIn("Loaded transcript match", result["toasts"])
        self.assertEqual(result["pulses"], ["target"])
        self.assertEqual(result["invalidate"], 1)
        self.assertEqual(result["finishes"], 1)

    def test_load_cursor_window_401_triggers_auth_loss(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            const err = new Error("unauth"); err.status = 401;
            h.setApiHandler(() => Promise.reject(err));
            const ok = await h.controller.loadChatSearchCursorWindow("LDC", { targetHistoryCursor: "" });
            globalThis.__result = { ok, authLoss: h.authLossCalls.length, finishes: h.olderLoadFinishes.length };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["authLoss"], 1)
        self.assertEqual(result["finishes"], 1)

    def test_load_cursor_window_409_refreshes_session(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            const err = new Error("rebind"); err.status = 409;
            h.setApiHandler(() => Promise.reject(err));
            const ok = await h.controller.loadChatSearchCursorWindow("LDC", { targetHistoryCursor: "" });
            globalThis.__result = { ok, openSession: h.openSessionCalls };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["openSession"], [{"sid": "sid-1", "opts": {"useCache": False}}])

    def test_load_cursor_window_generic_error_shows_older_error(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setApiHandler(() => Promise.reject(new Error("boom")));
            const ok = await h.controller.loadChatSearchCursorWindow("LDC", { targetHistoryCursor: "" });
            globalThis.__result = { ok, showOlderError: h.showOlderErrorCalls.length, finishes: h.olderLoadFinishes.length };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["showOlderError"], 1)
        self.assertEqual(result["finishes"], 1)

    def test_load_cursor_window_empty_cursor_returns_false(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            const ok = await h.controller.loadChatSearchCursorWindow("  ", { targetHistoryCursor: "" });
            globalThis.__result = { ok, apiCalls: h.apiCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["apiCalls"], 0)

    def test_load_cursor_window_empty_events_returns_false(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([]);
            h.controller.refreshLoaded({ jump: false, preserveCurrent: false });
            h.setApiHandler(() => Promise.resolve({ events: [], has_older: false }));
            const ok = await h.controller.loadChatSearchCursorWindow("LDC", { targetHistoryCursor: "" });
            globalThis.__result = { ok, finishes: h.olderLoadFinishes.length };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["ok"])
        self.assertEqual(result["finishes"], 1)

    # --- 12. event handlers + dispose ---

    def test_search_btn_toggles_open_close(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            const e = { _p: false, _s: false, preventDefault() { this._p = true; }, stopPropagation() { this._s = true; } };
            h.dom.chatSearchBtn.onclick(e);
            const afterOpen = { open: h.controller.isOpen(), display: h.dom.chatSearchBar.style.display, p: e._p, s: e._s };
            h.dom.chatSearchBtn.onclick(e);
            const afterClose = { open: h.controller.isOpen(), display: h.dom.chatSearchBar.style.display };
            globalThis.__result = { afterOpen, afterClose };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["afterOpen"]["open"])
        self.assertEqual(result["afterOpen"]["display"], "flex")
        self.assertTrue(result["afterOpen"]["p"])
        self.assertTrue(result["afterOpen"]["s"])
        self.assertFalse(result["afterClose"]["open"])
        self.assertEqual(result["afterClose"]["display"], "none")

    def test_input_oninput_triggers_refresh(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle", { order: 1 })]);
            h.dom.chatSearchInput.oninput();
            globalThis.__result = { matches: h.controller.currentMatches().length, applyMarks: h.applyMarksCalls.length };
            """
        )
        result = run_node_json(js)
        self.assertEqual(result["matches"], 1)
        self.assertTrue(result["applyMarks"] >= 1)

    def test_input_escape_closes_and_enter_steps(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("");
            h.controller.open();
            const esc = { key: "Escape", _p: false, preventDefault() { this._p = true; } };
            h.dom.chatSearchInput.onkeydown(esc);
            const afterEsc = { open: h.controller.isOpen(), prevented: esc._p };
            h.controller.open();
            h.setInput("");
            const enter = { key: "Enter", shiftKey: false, _p: false, preventDefault() { this._p = true; } };
            h.dom.chatSearchInput.onkeydown(enter);
            // Allow the void step(...) microtask to flush.
            await new Promise((r) => setTimeout(r, 0));
            globalThis.__result = { afterEsc, enterPrevented: enter._p, toasts: h.toasts };
            """
        )
        result = run_node_json(js)
        self.assertFalse(result["afterEsc"]["open"])
        self.assertTrue(result["afterEsc"]["prevented"])
        self.assertTrue(result["enterPrevented"])
        self.assertIn("Enter a loaded-chat search", result["toasts"])

    def test_prev_next_close_handlers_delegate(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.reset();
            h.select("sid-1");
            h.setInput("needle");
            h.setRows([h.makeRow("r1", "needle a", { order: 1 }), h.makeRow("r2", "needle b", { order: 2 })]);
            const mk = () => ({ _p: false, _s: false, preventDefault() { this._p = true; }, stopPropagation() { this._s = true; } });
            const prevE = mk(); h.dom.chatSearchPrevBtn.onclick(prevE);
            await new Promise((r) => setTimeout(r, 0));
            const nextE = mk(); h.dom.chatSearchNextBtn.onclick(nextE);
            await new Promise((r) => setTimeout(r, 0));
            const closeE = mk(); h.dom.chatSearchCloseBtn.onclick(closeE);
            globalThis.__result = { prevPrevent: prevE._p, nextPrevent: nextE._p, closePrevent: closeE._p, openAfterClose: h.controller.isOpen() };
            """
        )
        result = run_node_json(js)
        self.assertTrue(result["prevPrevent"])
        self.assertTrue(result["nextPrevent"])
        self.assertTrue(result["closePrevent"])
        self.assertFalse(result["openAfterClose"])

    def test_dispose_clears_handlers(self) -> None:
        js = harness_script(
            """
            const h = globalThis.__harness;
            h.controller.dispose();
            globalThis.__result = {
              btn: h.dom.chatSearchBtn.onclick,
              input: h.dom.chatSearchInput.oninput,
              keydown: h.dom.chatSearchInput.onkeydown,
              prev: h.dom.chatSearchPrevBtn.onclick,
              next: h.dom.chatSearchNextBtn.onclick,
              close: h.dom.chatSearchCloseBtn.onclick,
            };
            """
        )
        result = run_node_json(js)
        for key in ("btn", "input", "keydown", "prev", "next", "close"):
            self.assertIsNone(result[key], key)

    # --- 13. app.js delegation / load-order ---

    def test_index_loads_chat_search_after_navigation_before_app_js(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("app_chat_search.js?v=__CODOXEAR_ASSET_VERSION__", source)
        self.assertLess(
            source.index("app_chat_navigation.js?v=__CODOXEAR_ASSET_VERSION__"),
            source.index("app_chat_search.js?v=__CODOXEAR_ASSET_VERSION__"),
        )
        self.assertLess(
            source.index("app_chat_search.js?v=__CODOXEAR_ASSET_VERSION__"),
            source.index("app.js?v=__CODOXEAR_ASSET_VERSION__"),
        )

    def test_app_js_delegates_search_to_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module_source = APP_CHAT_SEARCH_JS.read_text(encoding="utf-8")
        self.assertIn("chatSearchController = (function instantiateChatSearchController() {", source)
        self.assertIn("const codoxearChatSearch = window.CodoxearChatSearch;", source)
        self.assertIn("codoxearChatSearch.createChatSearchController({", source)
        self.assertIn("createLoadedChatSearchRuntime: codoxearTranscript.createLoadedChatSearchRuntime,", source)
        self.assertIn("createChatSearchAllRuntime: codoxearTranscript.createChatSearchAllRuntime,", source)
        self.assertIn("olderLoadRuntime,\n          });", source)
        self.assertIn("function openChatSearch() {\n          chatSearchController.open();\n        }", source)
        self.assertIn("function closeChatSearch() {\n          chatSearchController.close();\n        }", source)
        self.assertIn("function refreshLoadedChatSearch(options) {\n          chatSearchController.refreshLoaded(options);\n        }", source)
        self.assertIn("if (chatSearchController) chatSearchController.dispose();", source)
        # Inline bodies moved to the module.
        self.assertNotIn("function syncChatSearchStatus() {", source)
        self.assertNotIn("async function refreshAllChatSearchCount(query)", source)
        self.assertNotIn("async function loadNearestOlderChatSearchWindow()", source)
        self.assertNotIn("async function loadChatSearchCursorWindow(cursor,", source)
        self.assertIn("function syncChatSearchStatus() {", module_source)
        self.assertIn("async function refreshAllChatSearchCount(query)", module_source)
        self.assertIn("async function loadNearestOlderChatSearchWindow()", module_source)
        self.assertIn("async function loadChatSearchCursorWindow(cursor,", module_source)


if __name__ == "__main__":
    unittest.main()
