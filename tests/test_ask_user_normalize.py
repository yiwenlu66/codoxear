"""Claude AskUserQuestion normalization tests."""

import unittest

from codoxear.rollout_log import _claude_ask_user_questions


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

    def test_question_without_text_dropped(self):
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
