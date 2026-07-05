import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.broker_log_binding import cc_fallback_session_log
from codoxear.util import find_new_session_log


SID = "68690c2f-104b-aaaa-bbbb-ccccddddffff"


def _write_cc_log(path: Path, *, sid: str = SID, cwd: str | None = "/work", content: str = "PING") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict = {
        "type": "user",
        "sessionId": sid,
        "timestamp": "2026-07-04T00:00:00.000Z",
        "message": {"role": "user", "content": content},
    }
    if cwd is not None:
        row["cwd"] = cwd
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


class TestCcFallbackSessionLog(unittest.TestCase):
    def _fallback(self, sessions_dir: Path, *, cwd: str, preexisting: set[Path] | None = None, after_ts: float = 0.0):
        return cc_fallback_session_log(
            sessions_dir=sessions_dir,
            cwd=cwd,
            after_ts=after_ts,
            preexisting=preexisting or set(),
            exclude_paths=set(),
            find_new_session_log_func=find_new_session_log,
        )

    def test_binds_when_recorded_cwd_matches_broker_cwd(self) -> None:
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            log = _write_cc_log(sessions / "-work" / f"{SID}.jsonl", cwd="/work")
            result = self._fallback(sessions, cwd="/work")
            assert result is not None
            self.assertEqual(result.log_path, log)
            self.assertEqual(result.session_id, SID)

    def test_binds_when_recorded_cwd_field_is_absent(self) -> None:
        # A third-party gateway (or schema drift) may omit the per-row cwd field.
        # The transcript still exists on disk and must be bound.
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            log = _write_cc_log(sessions / "-work" / f"{SID}.jsonl", cwd=None)
            result = self._fallback(sessions, cwd="/work")
            assert result is not None
            self.assertEqual(result.log_path, log)

    def test_binds_when_recorded_cwd_diverges_from_broker_cwd(self) -> None:
        # A login shell profile/rc may `cd` before exec, so Claude records a cwd
        # that differs from the broker's --cwd. This is the reported failure mode
        # (PONG written to disk, transcript_state stays pending_bind).
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            log = _write_cc_log(sessions / "-home" / f"{SID}.jsonl", cwd=str(Path.home()))
            result = self._fallback(sessions, cwd="/work")
            assert result is not None
            self.assertEqual(result.log_path, log)

    def test_does_not_bind_ambiguous_concurrent_fresh_logs(self) -> None:
        # The unscoped retry must refuse to guess when two fresh logs exist.
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            _write_cc_log(sessions / "-work-a" / "a.jsonl", sid="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", cwd="/work-a")
            _write_cc_log(sessions / "-work-b" / "b.jsonl", sid="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", cwd="/work-b")
            result = self._fallback(sessions, cwd="/work")
            self.assertIsNone(result)

    def test_does_not_bind_when_only_candidate_is_excluded_as_preexisting(self) -> None:
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            log = _write_cc_log(sessions / "-work" / f"{SID}.jsonl", cwd="/work")
            result = self._fallback(sessions, cwd="/work", preexisting={log})
            self.assertIsNone(result)

    def test_current_find_new_session_log_still_misses_divergent_cwd(self) -> None:
        # Documents the gap the fallback closes: the bare cwd-scoped lookup
        # returns None for a divergent-cwd log, so without the unscoped retry the
        # broker would never bind it.
        with TemporaryDirectory() as td:
            sessions = Path(td) / ".claude" / "projects"
            _write_cc_log(sessions / "-home" / f"{SID}.jsonl", cwd=str(Path.home()))
            found = find_new_session_log(
                sessions_dir=sessions,
                agent_backend="cc",
                cwd="/work",
                after_ts=0.0,
                preexisting=set(),
                timeout_s=0.0,
            )
            self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
