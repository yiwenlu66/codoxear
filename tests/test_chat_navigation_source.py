import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


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

    def test_jump_logic_is_loaded_user_rows_only(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('function loadedUserMessageRows() {', source)
        self.assertIn('row.dataset.role === "user"', source)
        self.assertIn('function jumpToLoadedUserMessage(direction)', source)
        self.assertIn('setToast("No loaded user messages")', source)
        self.assertIn('setToast("At first loaded user message")', source)
        self.assertIn('setToast("At last loaded user message")', source)
        self.assertIn('target.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" })', source)

    def test_jump_target_has_temporary_pulse_style(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn('row.classList.add("nav-pulse")', source)
        self.assertIn(".msg-row.nav-pulse .msg", css)
        self.assertIn("@keyframes navPulse", css)

    def test_loaded_chat_search_is_rendered_row_scoped_with_all_transcript_count(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "chatSearchBtn"', source)
        self.assertIn('title: "Search loaded messages"', source)
        self.assertIn('placeholder: "Search loaded chat"', source)
        self.assertIn('function refreshLoadedChatSearch', source)
        self.assertIn('chatSearchMatches = renderedMessageRows().filter', source)
        self.assertIn('rowSearchText(row).toLowerCase().includes(query)', source)
        self.assertIn('setToast(chatSearchQuery ? "No loaded matches" : "Enter a loaded-chat search")', source)
        self.assertIn('function refreshAllChatSearchCount(query)', source)
        self.assertIn('messages/search?q=${encodeURIComponent(cleanQuery)}&limit=0', source)
        self.assertIn('`${total ? chatSearchIndex + 1 : 0}/${total} loaded${allSuffix}`', source)
        self.assertIn('function loadOlderUntilChatSearchMatch({ boundaryMatch = null, focus = "first" } = {})', source)
        self.assertIn('const maxPages = 12;', source)
        self.assertIn('const loaded = await loadOlderMessages({ auto: false, cancelOnScroll: false });', source)
        self.assertIn('refreshLoadedChatSearch({ jump: false, preserveCurrent: false });', source)
        self.assertIn('const boundaryIndex = chatSearchMatches.indexOf(boundaryMatch);', source)
        self.assertIn('const unloadedTranscriptMatches = Number.isFinite(chatSearchAllCount) ? chatSearchAllCount > chatSearchMatches.length : true;', source)
        self.assertIn('const canLoadOlderMatches = Boolean(chatSearchQuery && unloadedTranscriptMatches && hasOlder);', source)
        self.assertIn('const startIndex = chatSearchIndex;', source)
        self.assertIn('const atForwardWrap = delta > 0 && startIndex >= chatSearchMatches.length - 1;', source)
        self.assertIn('boundaryMatch: chatSearchMatches[0],', source)
        self.assertIn('let olderLoadCancelOnScroll = true;', source)
        self.assertIn('async function loadOlderMessages({ auto = false, cancelOnScroll = true } = {})', source)
        self.assertIn('if (loadingOlder && olderLoadCancelOnScroll && cur > OLDER_CANCEL_PX) invalidateOlderLoad();', source)
        self.assertIn('void stepChatSearch(1);', source)

    def test_chat_search_has_safe_keyboard_shortcut(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('function chatNavigationShortcutBlocked(target) {', source)
        self.assertIn('if (!selected) return true;', source)
        self.assertIn('if (isTextEntryElement(target)) return true;', source)
        self.assertIn('if (document.body.classList.contains("sidebar-open")) return true;', source)
        self.assertIn('return modalIsolationTargets.some(isModalTargetOpen);', source)
        self.assertIn('function chatSearchShortcutBlocked(target) {', source)
        self.assertIn('return chatNavigationShortcutBlocked(target);', source)
        self.assertIn('if (e.defaultPrevented) return;', source)
        self.assertIn('if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {', source)
        self.assertIn('if (chatSearchShortcutBlocked(e.target)) return;', source)
        self.assertIn('openChatSearch();', source)
        self.assertIn('Use <b>/</b> to search the loaded chat; Previous/Next can page older history when the transcript count shows more matches.', source)

    def test_loaded_user_turn_navigation_has_keyboard_shortcut(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {', source)
        self.assertIn('if (chatNavigationShortcutBlocked(e.target)) return;', source)
        self.assertIn('jumpToLoadedUserMessage(e.key === "ArrowUp" ? -1 : 1);', source)
        self.assertIn('Use <b>Alt+↑</b>/<b>Alt+↓</b> to jump between loaded user messages without opening another panel.', source)

    def test_loaded_chat_search_has_compact_in_flow_styles(self) -> None:
        css = APP_CSS.read_text(encoding="utf-8")
        self.assertIn(".chatSearchBar", css)
        self.assertIn(".chatSearchInput", css)
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
