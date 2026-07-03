import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.agent_backend import get_agent_backend
from codoxear.agent_backend import infer_agent_backend_from_log_path
from codoxear.agent_backend import normalize_agent_backend


class TestCcBackendRegistration(unittest.TestCase):
    def test_cc_backend_registered(self) -> None:
        backend = get_agent_backend("cc")
        self.assertEqual(normalize_agent_backend("cc"), "cc")
        self.assertEqual(backend.default_bin, "claude")
        self.assertEqual(backend.bin_env_var, "CLAUDE_BIN")
        self.assertEqual(backend.home_env_var, "CLAUDE_CONFIG_DIR")
        self.assertEqual(backend.sessions_relpath, ("projects",))

    def test_infer_cc_log_path_excludes_subagent_logs(self) -> None:
        main = Path("/home/alice/.claude/projects/-repo/11111111-2222-3333-4444-555555555555.jsonl")
        sub = Path("/home/alice/.claude/projects/-repo/11111111-2222-3333-4444-555555555555/subagents/agent-abc.jsonl")
        self.assertEqual(infer_agent_backend_from_log_path(main), "cc")
        self.assertIsNone(infer_agent_backend_from_log_path(sub))

    def test_infer_cc_log_path_respects_custom_config_dir(self) -> None:
        with TemporaryDirectory() as td, patch.dict("os.environ", {"CLAUDE_CONFIG_DIR": td}):
            path = Path(td) / "projects" / "-repo" / "11111111-2222-3333-4444-555555555555.jsonl"
            self.assertEqual(infer_agent_backend_from_log_path(path), "cc")

    def test_backend_adapters_own_run_settings_extraction(self) -> None:
        calls = []

        def clean(value):
            return value.strip() if isinstance(value, str) and value.strip() else None

        def effort(value):
            return value.strip().lower() if isinstance(value, str) and value.strip() else None

        codex_settings = get_agent_backend("codex").read_run_settings_from_log(
            Path("/tmp/codex.jsonl"),
            read_pi_run_settings=lambda path: calls.append(("pi", path)) or ("pi-provider", "pi-model", "medium"),
            read_cc_run_settings=lambda path: calls.append(("cc", path)) or (None, "cc-model", "high"),
            read_session_meta_or_none_func=lambda path, **kwargs: {"model_provider": " openai ", "model": "", "reasoning_effort": ""},
            clean_optional_text=clean,
            display_reasoning_effort=effort,
            find_latest_turn_context=lambda path, **kwargs: {"model": " gpt-5.4 ", "effort": "HIGH"},
        )
        self.assertEqual(codex_settings, ("openai", "gpt-5.4", "high"))
        self.assertEqual(calls, [])

        pi_settings = get_agent_backend("pi").read_run_settings_from_log(
            Path("/tmp/pi.jsonl"),
            read_pi_run_settings=lambda path: ("macaron", "pi-model", "medium"),
            read_cc_run_settings=lambda path: self.fail("cc reader should not handle pi settings"),
            read_session_meta_or_none_func=lambda path, **kwargs: self.fail("codex meta should not handle pi settings"),
            clean_optional_text=clean,
            display_reasoning_effort=effort,
            find_latest_turn_context=lambda path, **kwargs: self.fail("codex turn context should not handle pi settings"),
        )
        self.assertEqual(pi_settings, ("macaron", "pi-model", "medium"))

        cc_settings = get_agent_backend("cc").read_run_settings_from_log(
            Path("/tmp/cc.jsonl"),
            read_pi_run_settings=lambda path: self.fail("pi reader should not handle cc settings"),
            read_cc_run_settings=lambda path: (None, "sonnet", "max"),
            read_session_meta_or_none_func=lambda path, **kwargs: self.fail("codex meta should not handle cc settings"),
            clean_optional_text=clean,
            display_reasoning_effort=effort,
            find_latest_turn_context=lambda path, **kwargs: self.fail("codex turn context should not handle cc settings"),
        )
        self.assertEqual(cc_settings, (None, "sonnet", "max"))

    def test_backend_adapters_own_row_busy_predicates(self) -> None:
        pi = get_agent_backend("pi")
        cc = get_agent_backend("cc")
        self.assertTrue(pi.message_keeps_turn_busy({"type": "message", "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "tool-a"}]}}))
        self.assertTrue(pi.message_keeps_turn_busy({"type": "message", "message": {"role": "assistant", "content": [{"type": "thinking", "text": "..."}]}}))
        self.assertTrue(pi.message_keeps_turn_busy({"type": "message", "message": {"role": "toolResult", "toolCallId": "tool-a"}}))
        self.assertFalse(pi.message_keeps_turn_busy({"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}))
        self.assertTrue(cc.message_keeps_turn_busy({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-a"}]}}))
        self.assertTrue(cc.message_keeps_turn_busy({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "text": "..."}]}}))
        self.assertTrue(cc.message_keeps_turn_busy({"type": "user", "toolUseResult": {"tool_use_id": "tool-a"}}))
        self.assertFalse(cc.message_keeps_turn_busy({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}))

    def test_backend_adapters_own_log_path_and_session_id_semantics(self) -> None:
        codex = get_agent_backend("codex")
        pi = get_agent_backend("pi")
        cc = get_agent_backend("cc")
        codex_log = Path("/home/alice/.codex/sessions/rollout-2026-07-03T00-00-00-11111111-2222-3333-4444-555555555555.jsonl")
        self.assertTrue(codex.is_session_log_path(codex_log))
        self.assertEqual(codex.log_glob_pattern(), "rollout-*.jsonl")
        self.assertEqual(codex.session_id_from_log_path(codex_log), "11111111-2222-3333-4444-555555555555")
        self.assertTrue(codex.log_matches_session_id(codex_log, "11111111-2222-3333-4444-555555555555"))

        with TemporaryDirectory() as td:
            root = Path(td)
            pi_dir = root / ".pi" / "agent" / "sessions"
            pi_dir.mkdir(parents=True)
            pi_log = pi_dir / "pi-session.jsonl"
            pi_log.write_text(json.dumps({"type": "session", "id": "pi-session", "cwd": "/repo"}) + "\n", encoding="utf-8")
            cc_dir = root / ".claude" / "projects" / "-repo"
            cc_dir.mkdir(parents=True)
            cc_log = cc_dir / "cc-session.jsonl"
            cc_log.write_text(json.dumps({"sessionId": "cc-session", "type": "system", "cwd": "/repo"}) + "\n", encoding="utf-8")
            cc_sub = cc_dir / "subagents" / "agent.jsonl"
            cc_sub.parent.mkdir()
            cc_sub.write_text("{}\n", encoding="utf-8")

            self.assertTrue(pi.is_session_log_path(pi_log, sessions_dir=pi_dir))
            self.assertEqual(pi.session_id_from_log_path(pi_log), "pi-session")
            self.assertTrue(pi.log_matches_session_id(pi_log, "pi-session"))
            self.assertTrue(cc.is_session_log_path(cc_log, sessions_dir=root / ".claude" / "projects"))
            self.assertFalse(cc.is_session_log_path(cc_sub, sessions_dir=root / ".claude" / "projects"))
            self.assertEqual(cc.session_id_from_log_path(cc_log), "cc-session")
            self.assertTrue(cc.log_matches_session_id(cc_log, "cc-session"))


if __name__ == "__main__":
    unittest.main()
