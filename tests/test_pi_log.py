import json
import tempfile
import unittest
from pathlib import Path

from codoxear.pi_log import pi_model_context_window
from codoxear.pi_log import read_pi_run_settings


class TestPiLogRunSettings(unittest.TestCase):
    def test_pi_model_context_window_falls_back_to_unique_model_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            models_path = Path(td) / "models.json"
            models_path.write_text(json.dumps({
                "providers": {
                    "macaron": {"models": [{"id": "gpt-5.4", "contextWindow": 1000000}]},
                    "openai": {"models": []},
                }
            }), encoding="utf-8")

            self.assertEqual(pi_model_context_window("openai", "gpt-5.4", models_path=models_path), 1000000)

    def test_read_pi_run_settings_recovers_early_model_events_from_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "session",
                    "version": 3,
                    "id": "sess-1",
                    "timestamp": "2026-04-16T14:45:09.869Z",
                    "cwd": "/tmp/project",
                }) + "\n")
                f.write(json.dumps({
                    "type": "model_change",
                    "provider": "openai",
                    "modelId": "gpt-5.4",
                }) + "\n")
                f.write(json.dumps({
                    "type": "thinking_level_change",
                    "thinkingLevel": "high",
                }) + "\n")
                filler = "x" * (9 * 1024 * 1024)
                f.write(json.dumps({
                    "type": "message",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": filler}]},
                }) + "\n")

            self.assertEqual(read_pi_run_settings(path), ("openai", "gpt-5.4", "high"))
