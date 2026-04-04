import json
import unittest

from tests.pi_fixtures import build_session_history
from tests.pi_fixtures import pi_persisted_session_file
from tests.pi_fixtures import pi_rpc_request_payloads
from tests.pi_fixtures import pi_persisted_session_entries
from tests.pi_fixtures import pi_rpc_response_lines
from tests.pi_fixtures import pi_runtime_session_file
from tests.pi_fixtures import pi_stream_events


class TestPiFixtures(unittest.TestCase):
    def test_rpc_request_fixture_covers_prompt_abort_and_get_state(self) -> None:
        requests = pi_rpc_request_payloads()

        self.assertEqual(set(requests), {"abort", "get_state", "prompt"})

        prompt = requests["prompt"]
        abort = requests["abort"]
        state = requests["get_state"]

        self.assertEqual(prompt["id"], "cmd-prompt-1")
        self.assertEqual(prompt["type"], "prompt")
        self.assertEqual(prompt["message"], "Summarize the current repository state.")
        self.assertEqual(abort["type"], "abort")
        self.assertEqual(state["type"], "get_state")

    def test_rpc_fixture_covers_prompt_abort_and_get_state(self) -> None:
        responses = pi_rpc_response_lines()

        self.assertEqual(set(responses), {"abort", "get_state", "prompt"})
        self.assertTrue(all(line.endswith("\n") for line in responses.values()))
        prompt = json.loads(responses["prompt"])
        abort = json.loads(responses["abort"])
        state = json.loads(responses["get_state"])

        self.assertEqual(prompt["type"], "response")
        self.assertEqual(prompt["id"], "cmd-prompt-1")
        self.assertTrue(prompt["success"])
        self.assertEqual(prompt["command"], "prompt")
        self.assertFalse(prompt["data"]["queued"])
        self.assertTrue(abort["success"])
        self.assertEqual(abort["command"], "abort")
        self.assertFalse(state["data"]["isStreaming"])
        self.assertEqual(state["data"]["messageCount"], 2)

    def test_persisted_session_fixture_uses_pi_like_session_file_shape(self) -> None:
        persisted = pi_persisted_session_file()
        header = persisted[0]
        user_entry = persisted[1]
        assistant_entry = persisted[2]

        self.assertEqual(header["type"], "session")
        self.assertEqual(header["session_id"], "pi-session-001")
        self.assertEqual(header["cwd"], "/workspace/codoxear")
        self.assertEqual(user_entry["type"], "message")
        self.assertEqual(user_entry["payload"]["type"], "message")
        self.assertEqual(user_entry["payload"]["role"], "user")
        self.assertEqual(user_entry["payload"]["content"][0]["text"], "Summarize the current repository state.")
        self.assertEqual(assistant_entry["payload"]["role"], "assistant")
        self.assertIn("Codoxear serves a browser UI", assistant_entry["payload"]["content"][0]["text"])

    def test_session_fixture_can_replay_user_and_assistant_history(self) -> None:
        events = pi_stream_events()
        entries = pi_persisted_session_entries()

        history = build_session_history(entries)
        turn_started = events[0]
        turn_completed = events[-1]
        persisted_user = entries[0]
        persisted_assistant = entries[1]

        self.assertEqual([event["type"] for event in events], ["turn.started", "message.delta", "tool.started", "turn.completed"])
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[0]["text"], "Summarize the current repository state.")
        self.assertIn("Codoxear serves a browser UI", history[1]["text"])
        self.assertEqual(persisted_user["payload"]["turn_id"], turn_started["turn_id"])
        self.assertEqual(persisted_assistant["payload"]["turn_id"], turn_completed["turn_id"])
        self.assertEqual(persisted_user["payload"]["source_event"], turn_started["type"])
        self.assertEqual(persisted_assistant["payload"]["source_event"], turn_completed["type"])
        self.assertEqual(history[0], {"role": persisted_user["payload"]["role"], "text": persisted_user["payload"]["content"][0]["text"]})
        self.assertEqual(history[1], {"role": persisted_assistant["payload"]["role"], "text": persisted_assistant["payload"]["content"][0]["text"]})
        self.assertEqual(persisted_assistant["payload"]["content"][0]["text"], turn_completed["text"])

    def test_runtime_session_fixture_matches_live_pi_message_shape(self) -> None:
        runtime = pi_runtime_session_file()

        self.assertEqual(runtime[0]["type"], "session")
        self.assertEqual(runtime[1]["type"], "message")
        self.assertEqual(runtime[1]["message"]["role"], "user")
        self.assertEqual(runtime[1]["message"]["content"][0]["type"], "text")
        self.assertEqual(runtime[2]["message"]["role"], "assistant")
        self.assertIn("Codoxear serves a browser UI", runtime[2]["message"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
