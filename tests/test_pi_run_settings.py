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


def test_pi_run_settings_finds_change_beyond_any_tail_window() -> None:
    # The v3 session header carries no model baseline, so a bounded tail scan
    # of a long-lived session's log misses a model_change written early on and
    # silently yields None (the UI then falls back to the stale launch model).
    # Production replay must reach the change no matter how far back it sits.
    with TemporaryDirectory() as td:
        path = Path(td) / "session.jsonl"
        padding = _row(
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "x" * 1024}]}}
        )
        with path.open("w", encoding="utf-8") as f:
            f.write(_row({"type": "session", "version": 3, "id": "pi-session", "cwd": "/tmp"}))
            f.write(_row({"type": "model_change", "provider": "terminal-provider", "modelId": "terminal-model"}))
            f.write(_row({"type": "thinking_level_change", "thinkingLevel": "xhigh"}))
            written = path.stat().st_size
            while written < 33 * 1024 * 1024:
                f.write(padding)
                written += len(padding)

        assert read_pi_run_settings(path) == ("terminal-provider", "terminal-model", "xhigh")
        # A diagnostic caller with an explicit bound accepts the miss.
        assert read_pi_run_settings(path, max_scan_bytes=1024 * 1024) == (None, None, None)
