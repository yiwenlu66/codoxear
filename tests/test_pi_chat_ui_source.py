import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class TestPiChatUiSource(unittest.TestCase):
    def test_pi_event_stream_state_and_helper_are_removed(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertNotIn('let piMessageView = "conversation";', source)
        self.assertNotIn("function sessionMessageView(s) {", source)

    def test_pi_view_toggle_is_removed(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function refreshSessions() {")
        end = source.index("function appendEvent(ev) {", start)
        refresh_block = source[start:end]

        self.assertNotIn('text: "Event Stream"', source)
        self.assertNotIn('class: "pi-view-btn', source)
        self.assertNotIn('id: "piViewToggle"', source)
        self.assertNotIn("syncPiViewToggle()", refresh_block)

    def test_render_pipeline_handles_reasoning_and_todo_rows_without_pi_event_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")

        self.assertIn('if (ev.type === "reasoning") {', source)
        self.assertIn('if (ev.type === "todo_snapshot") {', source)
        self.assertNotIn('if (ev.type === "pi_event") {', source)
        self.assertNotIn('pi_model_change', source)
        self.assertNotIn('pi_thinking_level_change', source)

    def test_set_typing_clears_last_known_tool_when_backend_reports_none(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function setTyping(show, toolName) {")
        end = source.index("function isNearBottom() {", start)
        block = source[start:end]

        self.assertIn("if (toolName === null) lastKnownTool = null;", block)

    def test_append_event_clears_assistant_dedupe_after_tool_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function appendEvent(ev) {")
        end = source.index("function prependOlderEvents(allEvents, { preserveViewport = false } = {}) {", start)
        block = source[start:end]

        self.assertIn('if (ev.type === "tool") {', block)
        self.assertIn('lastAssistantKey = "";', block)
        self.assertIn('if (ev.type === "subagent") {', block)

    def test_chat_pipeline_supports_tool_result_rows(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        block_start = source.index("function makeToolResultRow(ev) {")
        block_end = source.index("function makeSubagentRow(ev) {", block_start)
        block = source[block_start:block_end]

        self.assertIn('function makeToolResultRow(ev) {', source)
        self.assertIn('if (ev.type === "tool_result") {', source)
        self.assertIn('return `tool_result|${Math.round(ts * 1000)}|${ev.name || ""}|${text}`;', source)
        self.assertIn('if (ev.type === "tool_result" && typeof ev.name === "string") {', source)
        self.assertIn('void upgradeCandidateFileRefs(md);', block)


if __name__ == "__main__":
    unittest.main()
