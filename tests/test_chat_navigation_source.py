import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"
APP_CHAT_NAVIGATION_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app_chat_navigation.js"
APP_CHAT_SEARCH_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app_chat_search.js"
APP_DISPLAY_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app_display.js"
APP_MESSAGE_ROWS_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app_message_rows.js"
APP_TRANSCRIPT_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app_transcript.js"


class TestChatNavigationSource(unittest.TestCase):
    def test_loaded_user_message_jump_buttons_live_in_chat_nav_rail(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "prevUserBtn"', source)
        self.assertIn('title: "Previous user message"', source)
        self.assertIn('id: "nextUserBtn"', source)
        self.assertIn('title: "Next user message"', source)
        self.assertIn('const chatNavRail = el("div", { class: "chatNavRail"', source)
        self.assertIn("chatSearchBtn,\n          prevUserBtn,\n          nextUserBtn", source)
        topbar_start = source.index('const topbar = el("div", { class: "topbar" }')
        topbar_end = source.index('const form = el("form"', topbar_start)
        topbar_block = source[topbar_start:topbar_end]
        self.assertNotIn("prevUserBtn", topbar_block)
        self.assertNotIn("nextUserBtn", topbar_block)
        self.assertNotIn("chatSearchBtn", topbar_block)

    def test_session_utilities_are_outside_topbar(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('const sessionContextBar = el("div", { class: "sessionContextBar"', source)
        self.assertIn("fileBtn,\n          copyConversationBtn,\n          diagBtn,\n          unattendedBtn", source)
        topbar_start = source.index('const topbar = el("div", { class: "topbar" }')
        topbar_end = source.index('const form = el("form"', topbar_start)
        topbar_block = source[topbar_start:topbar_end]
        for name in ["fileBtn", "copyConversationBtn", "diagBtn", "unattendedBtn"]:
            self.assertNotIn(name, topbar_block)
        self.assertIn("interruptBtn", topbar_block)

    def test_loaded_user_message_rows_helper_remains_in_app_js(self) -> None:
        # The message-row helpers (loadedUserMessageRows / loadedUserJumpTarget)
        # and the row-source implementation stay in app.js / app_message_rows.js.
        source = APP_JS.read_text(encoding="utf-8")
        row_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        self.assertIn('function loadedUserMessageRows() {', source)
        self.assertIn('return codoxearMessageRows.loadedUserMessageRows(chatInner);', source)
        self.assertIn('row.dataset.role === "user"', row_source)
        self.assertIn('function loadedUserJumpTarget(rows, direction, threshold)', row_source)
        self.assertIn('return codoxearMessageRows.loadedUserJumpTarget(rows, direction, threshold);', source)
        self.assertIn('row.dataset.role === "user"', row_source)

    def test_navigation_jumps_are_delegated_to_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module_source = APP_CHAT_NAVIGATION_JS.read_text(encoding="utf-8")
        # app.js instantiates the controller and wires the message-row helpers.
        self.assertIn("const chatNavigationController = (function instantiateChatNavigationController() {", source)
        self.assertIn('const codoxearChatNavigation = window.CodoxearChatNavigation;', source)
        self.assertIn("codoxearChatNavigation.createChatNavigationController({", source)
        self.assertIn("loadedUserMessageRows,", source)
        self.assertIn("loadedCopyMessageRows,", source)
        self.assertIn("loadedUserJumpTarget,", source)
        self.assertIn("loadedCopyJumpTarget,", source)
        self.assertIn("getScrollTop: () => chat.scrollTop,", source)
        self.assertIn("prefersReducedMotion,", source)
        self.assertIn("pulseNavigatedRow,", source)
        self.assertIn("openChatSearch,", source)
        self.assertIn("isTextEntryElement,", source)
        self.assertIn("modalIsolationTargets,", source)
        self.assertIn("isModalTargetOpen,", source)
        self.assertIn("addAppEvent,", source)
        # app.js thin wrappers delegate to the controller; the inline bodies
        # moved out.
        self.assertIn("function updateChatNavButtons() {\n          chatNavigationController.syncButtons();\n        }", source)
        self.assertIn("function jumpToLoadedUserMessage(direction) {\n          chatNavigationController.jumpToLoadedUserMessage(direction);\n        }", source)
        self.assertIn("function jumpToLoadedMessage(direction) {\n          chatNavigationController.jumpToLoadedMessage(direction);\n        }", source)
        # The controller owns the boundary toasts and scroll/pulse behavior.
        self.assertIn('setToast("No loaded user messages")', module_source)
        self.assertIn('"At first loaded user message"', module_source)
        self.assertIn('"At last loaded user message"', module_source)
        self.assertIn('target.scrollIntoView({ block: "start", behavior: scrollBehavior() })', module_source)

    def test_jump_target_has_temporary_pulse_style(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        # pulseNavigatedRow stays in app.js so the chat-search path shares a
        # single row-pulse authority.
        self.assertIn("function pulseNavigatedRow(row)", source)
        self.assertIn('row.classList.add("nav-pulse")', source)
        self.assertIn(".msg-row.nav-pulse .msg", css)
        self.assertIn("@keyframes navPulse", css)

    def test_visible_time_indicator_uses_first_visible_message(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('const chatTimeChip = el("div", { id: "chatTimeChip", class: "chatTimeChip", "aria-hidden": "true" });', source)
        self.assertIn("chatWrap.appendChild(chatTimeChip);", source)
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        self.assertIn("const transcriptScrollRuntime = codoxearTranscript.createTranscriptScrollRuntime({", source)
        self.assertIn("isSearchOpen: () => chatSearchController.isOpen(),", source)
        self.assertIn("firstVisibleMessageRow,", source)
        self.assertNotIn("let autoScroll =", source)
        self.assertNotIn("let renderedAtLiveTail =", source)
        self.assertNotIn("let lastScrollTop =", source)
        self.assertIn("function syncVisibleTimeIndicator() {", transcript_source)
        self.assertIn("function syncJumpButton() {", transcript_source)
        self.assertIn("isSearchOpen()", transcript_source)
        self.assertIn("firstVisibleMessageRow()", transcript_source)
        self.assertIn('Number(row.dataset && row.dataset.ts ? row.dataset.ts : "0")', transcript_source)
        self.assertIn('timeChip.textContent = text;', transcript_source)
        self.assertIn('timeChip.style.display = "inline-flex";', transcript_source)
        self.assertIn('timeChip.style.display = "none";', transcript_source)
        self.assertIn("scrollRuntime.syncJumpButton();", transcript_source)
        self.assertIn("afterDecorate: () => {\n            updateChatNavButtons();", source)
        chat_search_source = APP_CHAT_SEARCH_JS.read_text(encoding="utf-8")
        self.assertIn("loadedChatSearchRuntime.setLoadingOlder(false);\n      syncVisibleTimeIndicator();", chat_search_source)
        reset_block = source[source.index("function resetChatRenderState()") : source.index("function clearTranscriptDom()")]
        self.assertIn("transcriptScrollRuntime.syncVisibleTimeIndicator();", reset_block)
        self.assertIn(".chatTimeChip", css)
        self.assertIn("pointer-events: none;", css)
        mobile_block = css[css.index("@media (max-width: 520px)") :]
        self.assertIn(".chatTimeChip", mobile_block)
        self.assertIn("top: auto;", mobile_block)
        self.assertIn("bottom: 14px;", mobile_block)

    def test_loaded_chat_search_is_rendered_row_scoped_with_all_transcript_count(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
        module_source = APP_CHAT_SEARCH_JS.read_text(encoding="utf-8")
        transcript_source = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        # DOM construction for the search bar/controls stays in app.js.
        self.assertIn('id: "chatSearchBtn"', source)
        self.assertIn('title: "Search loaded messages"', source)
        self.assertIn('placeholder: "Search loaded chat"', source)
        self.assertIn('const chatSearchAllHintEl = el("span", { id: "chatSearchAllHint", class: "chatSearchAllHint", text: "" });', source)
        # The search status projection, runtime ownership, refresh, all-count
        # scheduling/fetching, step semantics, and older-window loading moved
        # into the CodoxearChatSearch controller module.
        self.assertIn("const CHAT_SEARCH_ALL_DEBOUNCE_MS = 300;", module_source)
        self.assertIn("const CHAT_SEARCH_ALL_COUNT_MAX = 1000;", module_source)
        self.assertIn("const loadedChatSearchRuntime = createLoadedChatSearchRuntime();", module_source)
        self.assertIn("const chatSearchAllRuntime = createChatSearchAllRuntime({", module_source)
        self.assertIn("function refreshLoaded({ jump = false, preserveCurrent = true, refreshAllCount = true } = {}) {", module_source)
        self.assertIn("const matches = renderedMessageRows().filter", module_source)
        self.assertIn("loadedChatSearchRuntime.setMatches(matches, { preserveCurrent });", module_source)
        self.assertIn("rowSearchText(row).toLowerCase().includes(query)", module_source)
        self.assertIn('setToast(state.query ? "No loaded matches" : "Enter a loaded-chat search")', module_source)
        self.assertIn("chatSearchInput.oninput = () => refreshLoaded({ jump: true, preserveCurrent: false });", module_source)
        self.assertIn("function scheduleAllChatSearchCount(query)", module_source)
        self.assertIn("chatSearchAllRuntime.schedule(cleanQuery, (scheduledQuery) => {", module_source)
        self.assertIn("void refreshAllChatSearchCount(scheduledQuery);", module_source)
        self.assertIn("if (refreshAllCount) scheduleAllChatSearchCount(query);", module_source)
        self.assertNotIn("void refreshAllChatSearchCount(query);", module_source)
        self.assertIn("function refreshAllChatSearchCount(query)", module_source)
        self.assertIn("messages/search?q=${encodeURIComponent(cleanQuery)}&limit=1&text_max=96&count_max=${CHAT_SEARCH_ALL_COUNT_MAX}", module_source)
        self.assertNotIn("let chatSearchAllHint = \"\";", module_source)
        self.assertNotIn("let chatSearchAllLoadCursor = \"\";", module_source)
        self.assertIn("function compactChatSearchSnippet(text, query, limit = 96)", source)
        self.assertIn("return codoxearDisplay.compactChatSearchSnippet(text, query, limit);", source)
        self.assertIn("function compactChatSearchSnippet(text, query, limit = 96)", display_source)
        self.assertIn("if (clean.length <= maxLen) return clean;", display_source)
        self.assertIn("function chatSearchTranscriptHint(match, query)", source)
        self.assertIn("return codoxearDisplay.chatSearchTranscriptHint(match, query);", source)
        self.assertIn("function chatSearchTranscriptHint(match, query)", display_source)
        self.assertIn("chatSearchAllRuntime.completeRequest(request, {", module_source)
        self.assertIn("hint: chatSearchTranscriptHint(firstMatch, cleanQuery),", module_source)
        self.assertIn('chatSearchAllHintEl.textContent = showAllHint ? `all: ${allState.hint}` : "";', module_source)
        self.assertIn("void step(1);", module_source)
        self.assertIn('`${total ? searchState.index + 1 : 0}/${total} loaded${allSuffix}`', module_source)
        self.assertIn('${allState.count}${allState.truncated ? "+" : ""} all', module_source)
        self.assertIn("function loadOlderUntilChatSearchMatch({ boundaryMatch = null, focus = \"first\" } = {})", module_source)
        self.assertIn("const maxPages = 12;", module_source)
        self.assertIn("const loaded = await loadOlderMessages({ auto: false, cancelOnScroll: false });", module_source)
        self.assertIn("refreshLoaded({ jump: false, preserveCurrent: false });", module_source)
        self.assertIn("const boundaryIndex = matches.indexOf(boundaryMatch);", module_source)
        self.assertIn("const allState = chatSearchAllRuntime.snapshot();", module_source)
        self.assertIn("const unloadedTranscriptMatches = Number.isFinite(allState.count) ? (allState.truncated || allState.count > state.matches.length) : true;", module_source)
        self.assertIn("const canLoadOlderMatches = Boolean(state.query && unloadedTranscriptMatches && hasOlderMessages());", module_source)
        self.assertIn("(allState.truncated || allState.count > total)", module_source)
        self.assertIn("(allState.truncated || allState.count > state.matches.length)", module_source)
        self.assertIn("async function loadNearestOlderChatSearchWindow()", module_source)
        self.assertIn("const boundaryCursor = oldestRenderedHistoryCursor();", module_source)
        self.assertIn("order=latest&before=${encodeURIComponent(boundaryCursor)}", module_source)
        self.assertIn('const targetHistoryCursor = match && typeof match.history_cursor === "string" ? match.history_cursor : "";', module_source)
        self.assertIn("return await loadChatSearchCursorWindow(cursor, { targetHistoryCursor });", module_source)
        self.assertIn("async function loadChatSearchCursorWindow(cursor, { targetHistoryCursor = \"\" } = {})", module_source)
        self.assertIn("function ensureChatSearchTargetRow(historyCursor)", module_source)
        self.assertIn("row.dataset.historyCursor === targetCursor", module_source)
        self.assertIn('target.dataset.searchForcedQuery = normalizeQuery(forcedQuery);', transcript_source)
        self.assertIn("row.dataset.searchForcedQuery === query || rowSearchText(row).toLowerCase().includes(query)", module_source)
        self.assertIn("const targetIndex = ensureChatSearchTargetRow(targetHistoryCursor);", module_source)
        self.assertIn("else if (currentMatches().length) focusChatSearchMatch(currentMatches().length - 1, { jump: true });", module_source)
        self.assertIn("if (!evs.length) return false;", module_source)
        self.assertIn("renderDetachedTranscriptWindow(evs, { hasMore: Boolean(data.has_older) })", module_source)
        self.assertIn("const jumped = await loadNearestOlderChatSearchWindow();", module_source)
        self.assertIn("const startIndex = state.index;", module_source)
        self.assertIn("const atForwardWrap = delta > 0 && startIndex >= state.matches.length - 1;", module_source)
        self.assertIn("boundaryMatch: state.matches[0],", module_source)
        # Transcript older-load authority stays in app.js (injected into the
        # search controller).
        self.assertNotIn("let olderLoadCancelOnScroll = true;", source)
        self.assertIn("const olderLoadRuntime = codoxearTranscript.createOlderLoadRuntime({", source)
        self.assertIn("async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {})", source)
        self.assertIn("cancelOlderLoad: invalidateOlderLoad,", source)
        self.assertIn('if (shouldCancelOlderLoad() && cur > olderCancelPx) cancelOlderLoad();', transcript_source)
        # app.js delegates to the search controller through thin wrappers and
        # the instantiation block.
        self.assertIn("chatSearchController = (function instantiateChatSearchController() {", source)
        self.assertIn("const codoxearChatSearch = window.CodoxearChatSearch;", source)
        self.assertIn("codoxearChatSearch.createChatSearchController({", source)
        self.assertIn("createLoadedChatSearchRuntime: codoxearTranscript.createLoadedChatSearchRuntime,", source)
        self.assertIn("createChatSearchAllRuntime: codoxearTranscript.createChatSearchAllRuntime,", source)
        self.assertIn("function openChatSearch() {\n          chatSearchController.open();\n        }", source)
        self.assertIn("function closeChatSearch() {\n          chatSearchController.close();\n        }", source)
        self.assertIn("function refreshLoadedChatSearch(options) {\n          chatSearchController.refreshLoaded(options);\n        }", source)
        self.assertIn("function stepChatSearch(delta) {\n          return chatSearchController.step(delta);\n        }", source)
        # The inline search bodies are gone from app.js.
        self.assertNotIn("function syncChatSearchStatus() {", source)
        self.assertNotIn("async function refreshAllChatSearchCount(query)", source)
        self.assertNotIn("async function loadNearestOlderChatSearchWindow()", source)
        self.assertNotIn("async function loadChatSearchCursorWindow(cursor,", source)
        self.assertNotIn("const chatSearchAllRuntime = codoxearTranscript.createChatSearchAllRuntime({", source)

    def test_message_copy_buttons_use_roving_tab_stop(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        row_source = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertNotIn('let activeMessageCopyRow = null;', source)
        self.assertIn('const messageCopyNavigationRuntime = codoxearMessageRows.createMessageCopyNavigationRuntime({ root: chatInner });', source)
        self.assertIn('function syncMessageCopyTabStops()', source)
        self.assertIn('function loadedCopyMessageRows()', source)
        self.assertIn('btn.tabIndex = active ? 0 : -1;', row_source)
        self.assertIn('btn.disabled = !active;', row_source)
        self.assertIn('if (active) btn.removeAttribute("aria-hidden");', row_source)
        self.assertIn('else btn.setAttribute("aria-hidden", "true");', row_source)
        self.assertIn('tabindex: "-1",', row_source)
        self.assertIn('disabled: "true"', row_source)
        self.assertIn('"aria-hidden": "true"', row_source)
        self.assertIn('.msg-copy-btn[aria-hidden="true"],', css)
        self.assertIn('visibility: hidden;', css)
        self.assertIn('pointer-events: none;', css)
        self.assertIn('function activeElementIsMessageCopyButton()', source)
        self.assertIn('addAppEvent(chatInner, "pointerover"', source)
        self.assertIn('if (activeElementIsMessageCopyButton()) return;', source)
        self.assertIn('addAppEvent(chatInner, "focusin"', source)
        self.assertIn('setActiveMessageCopyRow(row, { focusCopy: activeElementIsMessageCopyButton() });', source)
        self.assertIn('function loadedCopyJumpTarget(rows, activeRow, direction, threshold)', row_source)
        self.assertIn('function createMessageCopyNavigationRuntime(options = {})', row_source)
        self.assertIn('return messageCopyNavigationRuntime.jumpTarget(rows, direction, threshold);', source)
        self.assertIn('function loadedCopyMessageRows()', source)
        self.assertIn('function jumpToLoadedMessage(direction)', source)

    def test_chat_navigation_shortcuts_are_owned_by_controller(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        module_source = APP_CHAT_NAVIGATION_JS.read_text(encoding="utf-8")
        # The blocking predicate and the keydown handler moved to the module.
        self.assertNotIn("function chatNavigationShortcutBlocked(target) {", source)
        self.assertNotIn("function chatSearchShortcutBlocked(target) {", source)
        self.assertIn("function chatNavigationShortcutBlocked(target) {", module_source)
        self.assertIn("function chatSearchShortcutBlocked(target) {", module_source)
        self.assertIn("if (!getSelected()) return true;", module_source)
        self.assertIn("if (isTextEntryElement(target)) return true;", module_source)
        self.assertIn("if (isSidebarOpen()) return true;", module_source)
        self.assertIn("return modalIsolationTargets.some(isModalTargetOpen);", module_source)
        self.assertIn('return chatNavigationShortcutBlocked(target);', module_source)
        self.assertIn("if (e.defaultPrevented) return;", module_source)
        self.assertIn('if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {', module_source)
        self.assertIn("if (chatSearchShortcutBlocked(e.target)) return;", module_source)
        self.assertIn("openChatSearch();", module_source)
        self.assertIn("if (e.altKey && e.shiftKey && !e.ctrlKey && !e.metaKey && (e.key === \"ArrowUp\" || e.key === \"ArrowDown\")) {", module_source)
        self.assertIn("jumpToLoadedMessage(e.key === \"ArrowUp\" ? -1 : 1);", module_source)
        self.assertIn("if (e.altKey && !e.shiftKey && !e.ctrlKey && !e.metaKey && (e.key === \"ArrowUp\" || e.key === \"ArrowDown\")) {", module_source)
        self.assertIn("jumpToLoadedUserMessage(e.key === \"ArrowUp\" ? -1 : 1);", module_source)
        # The help text remains in app.js.
        self.assertIn("Use <b>/</b> to search the loaded chat; Previous/Next can load an older matching window when the transcript count shows more matches.", source)
        self.assertIn("Use <b>Alt+↑</b>/<b>Alt+↓</b> to jump between loaded user messages.", source)
        self.assertIn("Alt+Shift+↑", source)

    def test_loaded_chat_search_has_compact_in_flow_styles(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".chatSearchBar", css)
        self.assertIn(".chatSearchInput", css)
        self.assertIn(".chatSearchAllHint", css)
        self.assertIn("display: none !important;", css[css.index("@media (max-width: 520px)"):])
        self.assertIn(".chatNavRail", css)
        self.assertIn(".msg-row.chat-search-current .msg", css)
        self.assertIn(".chatSearchBar {\n        order: 0;\n        flex: 0 0 auto;", css)
        search_block = css[css.index(".chatSearchBar {"):css.index("      .chatSearchInput", css.index(".chatSearchBar {"))]
        self.assertNotIn("position: absolute;", search_block)
        self.assertNotIn("transform: translateX", search_block)

    def test_chat_navigation_rail_is_in_layout_flow(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".chatWrap {\n        flex: 1;", css)
        self.assertIn("display: flex;\n        flex-direction: column;", css)
        self.assertIn(".chatNavRail {\n        display: none;\n        order: 0;", css)
        rail_block = css[css.index(".chatNavRail {"):css.index("      button {", css.index(".chatNavRail {"))]
        self.assertNotIn("position: absolute;", rail_block)
        self.assertNotIn("top: 10px;", rail_block)


if __name__ == "__main__":
    unittest.main()
