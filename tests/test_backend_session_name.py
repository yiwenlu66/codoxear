from __future__ import annotations

import json
from pathlib import Path

from codoxear.backend_session_name import backend_session_name_from_row
from codoxear.backend_session_name import read_last_backend_session_name


def test_pi_session_info_rows_carry_the_display_name() -> None:
    assert backend_session_name_from_row("pi", {"type": "session_info", "name": "  auth   refactor "}) == "auth refactor"
    # An empty name is an explicit clear, not "this row says nothing".
    assert backend_session_name_from_row("pi", {"type": "session_info", "name": ""}) == ""
    assert backend_session_name_from_row("pi", {"type": "session_info"}) == ""
    # Unrelated rows are silent.
    assert backend_session_name_from_row("pi", {"type": "model_change", "modelId": "x"}) is None
    assert backend_session_name_from_row("pi", {"type": "message", "name": "not a title"}) is None


def test_cc_custom_title_rows_carry_the_display_name() -> None:
    assert backend_session_name_from_row("cc", {"type": "custom-title", "customTitle": "deploy check"}) == "deploy check"
    assert backend_session_name_from_row("cc", {"type": "custom-title"}) == ""
    assert backend_session_name_from_row("cc", {"type": "summary", "summary": "x"}) is None


def test_codex_titles_are_not_in_the_rollout() -> None:
    # Codex keeps thread names in the app-server state, never in the transcript.
    assert backend_session_name_from_row("codex", {"type": "session_info", "name": "x"}) is None
    assert backend_session_name_from_row("codex", {"type": "custom-title", "customTitle": "x"}) is None
    assert backend_session_name_from_row("", {"type": "session_info", "name": "x"}) is None


def test_last_name_survives_junk_lines_and_a_trailing_clear(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "s1"}),
                "this line is not json",
                json.dumps({"type": "session_info", "name": "first"}),
                json.dumps({"type": "message", "role": "user", "content": "hi"}),
                json.dumps({"type": "session_info", "name": "second"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert read_last_backend_session_name(log, "pi") == "second"

    log.write_text(
        log.read_text(encoding="utf-8") + json.dumps({"type": "session_info", "name": ""}) + "\n",
        encoding="utf-8",
    )
    assert read_last_backend_session_name(log, "pi") == ""


def test_cc_transcript_scan_uses_custom_titles(tmp_path: Path) -> None:
    log = tmp_path / "cc.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "summary", "summary": "auto"}),
                json.dumps({"type": "custom-title", "customTitle": "codexear fix"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert read_last_backend_session_name(log, "cc") == "codexear fix"


def test_missing_log_is_not_an_error(tmp_path: Path) -> None:
    assert read_last_backend_session_name(tmp_path / "absent.jsonl", "pi") is None
