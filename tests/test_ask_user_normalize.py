"""Claude AskUserQuestion normalization tests."""

import unittest

from codoxear.rollout_log import _claude_ask_user_questions
from codoxear.rollout_log import _extract_chat_events


class ClaudeAskUserNormalizeTests(unittest.TestCase):
    def test_multiselect_maps_to_allow_multiple(self):
        qs = [{
            "header": "Auth",
            "multiSelect": True,
            "question": "Which?",
            "options": [{"label": "A", "description": "x"}, {"label": "B"}],
        }]
        out = _claude_ask_user_questions(qs)
        self.assertEqual(len(out), 1)
        q = out[0]
        self.assertTrue(q["allowMultiple"])
        self.assertEqual(q["header"], "Auth")
        self.assertEqual(q["backend"], "claude")
        self.assertEqual(q["question"], "Which?")
        self.assertEqual(
            q["options"],
            [{"label": "A", "description": "x"}, {"label": "B"}],
        )

    def test_single_select_default_when_multiselect_absent(self):
        out = _claude_ask_user_questions([{"question": "Where?", "options": [{"label": "C"}]}])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["allowMultiple"])
        self.assertNotIn("header", out[0])

    def test_multiselect_false_explicit(self):
        out = _claude_ask_user_questions([{"question": "Q", "multiSelect": False, "options": []}])
        self.assertFalse(out[0]["allowMultiple"])

    def test_question_falls_back_to_header(self):
        # Claude emits questions with only `header` (no `question`); the TUI
        # renders header as the title, so we keep the prompt instead of
        # dropping it. Live-confirmed: 31/31 such records carry a header.
        out = _claude_ask_user_questions([{"header": "Rollout turns", "options": [{"label": "A"}]}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["question"], "Rollout turns")
        self.assertEqual(out[0]["header"], "Rollout turns")

    def test_question_without_text_or_header_is_dropped(self):
        out = _claude_ask_user_questions([{"options": [{"label": "A"}]}, {"question": "ok", "options": []}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["question"], "ok")

    def test_non_list_returns_empty(self):
        self.assertEqual(_claude_ask_user_questions(None), [])
        self.assertEqual(_claude_ask_user_questions({}), [])

    def test_options_normalized_to_label_description(self):
        out = _claude_ask_user_questions([{
            "question": "Q",
            "options": ["plain", {"title": "T", "description": "d"}, {"label": ""}],
        }])
        self.assertEqual(
            out[0]["options"],
            [{"label": "plain"}, {"label": "T", "description": "d"}],
        )

    def test_rejected_missing_question_is_not_answerable(self):
        tool_id = "tooluse_missing_question"
        tool_use = {
            "type": "assistant",
            "timestamp": "2026-06-27T00:39:17.460Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "AskUserQuestion",
                    "input": {"questions": [{"header": "Run mode", "options": [{"label": "Safe"}]}]},
                }],
            },
        }
        tool_result = {
            "type": "user",
            "timestamp": "2026-06-27T00:39:17.462Z",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": True,
                    "content": (
                        "<tool_use_error>InputValidationError: AskUserQuestion failed: "
                        "The required parameter `questions[0].question` is missing</tool_use_error>"
                    ),
                }],
            },
        }

        events, _meta, _flags, _diag = _extract_chat_events([tool_use, tool_result])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["interactive"], "ask_user_rejected")
        self.assertEqual(events[0]["tool_use_id"], tool_id)
        self.assertEqual(events[0]["reason"], "missing_question")
        self.assertNotIn("questions", events[0])

    def test_rejected_tool_result_alone_supports_live_delta(self):
        tool_id = "tooluse_late_rejection"
        tool_result = {
            "type": "user",
            "timestamp": "2026-06-27T00:39:17.462Z",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": (
                        "InputValidationError: AskUserQuestion failed because the required parameter "
                        "questions[2].question is missing"
                    ),
                }],
            },
        }

        events, _meta, _flags, _diag = _extract_chat_events([tool_result])

        self.assertEqual([event["interactive"] for event in events], ["ask_user_rejected"])
        self.assertEqual(events[0]["tool_use_id"], tool_id)

    def test_well_formed_multi_question_prompt_remains_answerable(self):
        tool_use = {
            "type": "assistant",
            "timestamp": "2026-06-27T00:39:17.460Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "tooluse_valid_multi",
                    "name": "AskUserQuestion",
                    "input": {"questions": [
                        {"question": "First?", "header": "One", "options": [{"label": "A"}]},
                        {"question": "Second?", "header": "Two", "options": [{"label": "B"}]},
                    ]},
                }],
            },
        }

        events, _meta, _flags, _diag = _extract_chat_events([tool_use])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["interactive"], "ask_user_question")
        self.assertEqual([q["question"] for q in events[0]["questions"]], ["First?", "Second?"])

    def test_question_order_stable_for_isfinal_check(self):
        """The frontend uses qIdx === questions.length - 1 to detect the final
        question, which only works if the parser preserves input order. A
        4-question prompt must yield 4 outputs in the same order so the last
        input question is also the last output question.
        """
        qs = [
            {"question": "Q1", "header": "First", "options": [{"label": "a"}]},
            {"question": "Q2", "header": "Second", "options": [{"label": "b"}]},
            {"question": "Q3", "header": "Third", "options": [{"label": "c"}]},
            {"question": "Q4", "header": "Fourth", "options": [{"label": "d"}]},
        ]
        out = _claude_ask_user_questions(qs)
        self.assertEqual(len(out), 4)
        self.assertEqual([q["question"] for q in out], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual([q["header"] for q in out], ["First", "Second", "Third", "Fourth"])
        # The frontend's final-question detection must hold:
        self.assertEqual(out[-1]["question"], qs[-1]["question"])


if __name__ == "__main__":
    unittest.main()
