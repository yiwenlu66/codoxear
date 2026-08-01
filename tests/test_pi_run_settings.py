import json
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.pi_log import read_pi_run_settings


def _row(obj: dict) -> str:
    return json.dumps(obj) + "\n"


def test_pi_run_settings_replays_changes_older_than_legacy_tail_window() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "session.jsonl"
        path.write_text(
            "".join(
                [
                    _row(
                        {
                            "type": "session",
                            "id": "pi-session",
                            "provider": "launch-provider",
                            "modelId": "launch-model",
                            "thinkingLevel": "low",
                        }
                    ),
                    _row({"type": "model_change", "provider": "terminal-provider", "modelId": "terminal-model"}),
                    _row({"type": "thinking_level_change", "thinkingLevel": "xhigh"}),
                    *[_row({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "padding"}]}}) for _ in range(8)],
                ]
            ),
            encoding="utf-8",
        )

        # The explicit diagnostic bound demonstrates the conflict: launch
        # metadata wins when the authoritative changes fall outside the tail.
        assert read_pi_run_settings(path, max_scan_bytes=64) == ("launch-provider", "launch-model", "low")
        # The production default must replay the authoritative changes instead.
        assert read_pi_run_settings(path) == ("terminal-provider", "terminal-model", "xhigh")
