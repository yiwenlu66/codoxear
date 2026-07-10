"""Frontend source guards for rejected ask-user rendering."""

import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


class AskUserFrontendSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = APP_JS.read_text(encoding="utf-8")

    def test_rejected_prompt_has_non_interactive_notice(self) -> None:
        self.assertIn('ev.interactive === "ask_user_rejected"', self.source)
        self.assertIn("Prompt rejected by the agent — missing question text.", self.source)
        self.assertIn('text: "ask_user_rejected"', self.source)

    def test_late_rejection_replaces_answerable_prompt_by_tool_id(self) -> None:
        self.assertIn("function removeSupersededAskUserPrompt(toolUseId)", self.source)
        self.assertIn('row.dataset.toolUseId = ev.tool_use_id', self.source)
        self.assertIn('data-interactive="ask_user_question"', self.source)

    def test_history_batch_collapses_rejected_prompt(self) -> None:
        self.assertIn("function collapseRejectedAskUserEvents(events)", self.source)
        self.assertIn("for (const ev of collapseRejectedAskUserEvents(events || []))", self.source)
        self.assertIn("for (const ev of collapseRejectedAskUserEvents(allEvents))", self.source)


if __name__ == "__main__":
    unittest.main()
