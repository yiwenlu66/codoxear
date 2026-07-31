"""Deterministic proof of the interruption-transcript-outcome defect (current HEAD).

Goal: prove, from minimal synthetic logs, that a user turn that ends in an
interruption/abort leaves NO persistent transcript outcome row across the unit
normalizer, the transcript search, and the /api/messages (tail + search)
surfaces. Read-only: imports codoxear surfaces; writes only its own artifacts.

Three scenarios, exactly as specified by the task:
  A. Pi: user message then assistant stopReason:"aborted" with empty content.
  B. Pi: user message then assistant stopReason:"aborted" with partial text.
  C. Codex: event_msg user_message then event_msg turn_aborted.

Invariant under test: every sent turn must persistently render one of
answer / error / no-answer / interruption. The DEFECT is that for an
interruption the transcript renders only the user row (indistinguishable from
an ignored prompt) and, for Pi partial, the already-streamed partial text is
discarded.

Run from repo root:
    python3 .memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/interruption-outcome-defect/prove_interruption_defect.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_search
from codoxear.message_routes import handle_messages_tail
from codoxear.rollout_chat_events import _NO_RESPONSE_TEXT
from codoxear.rollout_jsonl import JsonlRecord
from codoxear.rollout_log import _extract_positioned_chat_events
from codoxear.rollout_log import _read_chat_tail_page
from codoxear.transcript_search import search_chat_log_bounded

SECRET = b"interrupt-proof-secret"

ARTIFACT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Log fixtures
# --------------------------------------------------------------------------
def pi_user_row(ts: float, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "ts": ts,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def pi_assistant_aborted_row(ts: float, *, partial_text: str | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if partial_text:
        content.append({"type": "text", "text": partial_text})
    return {
        "type": "message",
        "ts": ts,
        "message": {"role": "assistant", "stopReason": "aborted", "content": content},
    }


def codex_user_msg(ts: float, text: str) -> dict[str, Any]:
    return {"type": "event_msg", "ts": ts, "payload": {"type": "user_message", "message": text}}


def codex_turn_aborted(ts: float) -> dict[str, Any]:
    return {"type": "event_msg", "ts": ts, "payload": {"type": "turn_aborted"}}


SCENARIO_A = {  # Pi empty abort
    "name": "A-pi-empty-abort",
    "backend": "pi",
    "rows": [pi_user_row(1.0, "hello pi"), pi_assistant_aborted_row(2.0, partial_text=None)],
}
SCENARIO_B = {  # Pi partial-text abort
    "name": "B-pi-partial-abort",
    "backend": "pi",
    "partial": "I was halfway through",
    "rows": [pi_user_row(1.0, "hello pi partial"), pi_assistant_aborted_row(2.0, partial_text="I was halfway through")],
}
SCENARIO_C = {  # Codex turn_aborted
    "name": "C-codex-turn-aborted",
    "backend": "codex",
    "rows": [codex_user_msg(1.0, "hello codex"), codex_turn_aborted(2.0)],
}
SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def records_from_rows(rows: list[dict[str, Any]]) -> list[JsonlRecord]:
    """Build in-order JsonlRecords with byte offsets derived from serialized size."""
    out: list[JsonlRecord] = []
    offset = 0
    for row in rows:
        chunk = (json.dumps(row) + "\n").encode("utf-8")
        out.append(JsonlRecord(start=offset, end=offset + len(chunk), obj=row))
        offset += len(chunk)
    return out


# --------------------------------------------------------------------------
# Surface 1: unit normalization (in-memory records + disk-backed tail page)
# --------------------------------------------------------------------------
def unit_extract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _extract_positioned_chat_events(records_from_rows(rows))


def unit_tail(log_path: Path) -> list[dict[str, Any]]:
    events, _before, _after, _has_older = _read_chat_tail_page(log_path, limit=80)
    return events


def public_role_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal positioning keys; keep the user-visible projection."""
    KEEP = {"role", "text", "message_class", "message_id", "ts"}
    return [{k: ev.get(k) for k in KEEP} for ev in events]


# --------------------------------------------------------------------------
# Surface 2: transcript search (disk-backed)
# --------------------------------------------------------------------------
def search(log_path: Path, query: str) -> tuple[int, list[dict[str, Any]]]:
    count, matches, _trunc = search_chat_log_bounded(log_path, query, limit=20, max_line_bytes=4 * 1024 * 1024)
    return count, public_role_summary(matches)


# --------------------------------------------------------------------------
# Surface 3: /api/messages tail + search handlers
# --------------------------------------------------------------------------
from codoxear.session_model import Session  # noqa: E402


def make_session(td: str, log_path: Path, backend: str) -> Session:
    return Session(
        session_id="s1", thread_id="thread-1", broker_pid=1, codex_pid=1,
        agent_backend=backend, owned=False, start_ts=0.0, cwd=td,
        log_path=log_path, sock_path=Path(td) / "s1.sock",
    )


class Mgr:
    def __init__(self, session: Session) -> None:
        self._s = session

    def refresh_session_meta(self, _sid: str) -> None:
        return None

    def get_session(self, _sid: str) -> Session:
        return self._s

    def mark_log_delta(self, *a: Any, **k: Any) -> None:
        return None

    def _attach_notification_texts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return events


class FakeHandler:
    def _unauthorized(self) -> None:
        pass


def _deps_for(session: Session) -> tuple[MessageRouteDeps, list[tuple[int, dict[str, Any]]]]:
    responses: list[tuple[int, dict[str, Any]]] = []

    def json_response(_h: Any, status: int, payload: dict[str, Any]) -> None:
        responses.append((status, payload))

    def enc(*, kind: str, session: Any, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=SECRET)

    def dec(token: str, *, kind: str, session: Any) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=SECRET)

    def snap(_sid: str, _session: Any, **_kw: Any) -> tuple[dict[str, Any], bool, int, Any]:
        return {}, False, 0, None

    deps = MessageRouteDeps(
        require_auth=lambda _h: True,
        json_response=json_response,
        launch_attempt_transcript_for_session_id=lambda _sid: None,
        transcript_export_max_bytes=10 * 1024 * 1024,
        transcript_search_max_line_bytes=4 * 1024 * 1024,
        decode_message_cursor=dec,
        encode_message_cursor=enc,
        record_metric=lambda _n, _v: None,
        message_runtime_snapshot=snap,
    )
    return deps, responses


def api_tail(session: Session) -> dict[str, Any]:
    deps, responses = _deps_for(session)
    handle_messages_tail(FakeHandler(), session_id="s1", query="limit=80", manager=Mgr(session), deps=deps)
    return responses[0][1]


def api_search(session: Session, query: str) -> dict[str, Any]:
    deps, responses = _deps_for(session)
    handle_messages_search(FakeHandler(), session_id="s1", query=f"q={query}", manager=Mgr(session), deps=deps)
    return responses[0][1]


# --------------------------------------------------------------------------
# Verdict helpers
# --------------------------------------------------------------------------
INTERRUPTION_HINTS = ("interrupt", "abort", "stop", "cancel")  # words an interruption row would plausibly carry


def has_assistant_row(events: list[dict[str, Any]]) -> bool:
    return any(ev.get("role") == "assistant" for ev in events)


def rows_are_user_only(events: list[dict[str, Any]]) -> bool:
    roles = [ev.get("role") for ev in events]
    return roles == ["user"]


def classify(role_summary: list[dict[str, Any]], scenario: dict[str, Any]) -> str:
    """PASS = a distinct assistant outcome row exists; DEFECT = user-only / dropped."""
    if rows_are_user_only(role_summary):
        return "DEFECT (user-only; no interruption outcome row)"
    if scenario.get("name") == "B-pi-partial-abort":
        texts = [ev.get("text") for ev in role_summary if ev.get("role") == "assistant"]
        if scenario["partial"] in texts:
            return "PASS (partial text preserved as assistant row)"
    if has_assistant_row(role_summary):
        # Any persistent assistant row beyond the user row is a visible outcome.
        return "PASS (assistant outcome row present)"
    return "DEFECT (no assistant row)"


# --------------------------------------------------------------------------
# Main proof
# --------------------------------------------------------------------------
def run_scenario(scenario: dict[str, Any], td: Path) -> dict[str, Any]:
    name = scenario["name"]
    backend = scenario["backend"]
    log_path = td / f"{name}.jsonl"
    write_jsonl(log_path, scenario["rows"])

    # Surface 1: unit
    unit_events = public_role_summary(unit_extract(scenario["rows"]))
    tail_events = public_role_summary(unit_tail(log_path))
    # Surface 2: search
    user_text = scenario["rows"][0]["message"]["content"][0]["text"] if backend == "pi" else scenario["rows"][0]["payload"]["message"]
    ctrl_count, ctrl_matches = search(log_path, user_text)
    # Search for an interruption-worded outcome (should match if an interruption row existed)
    hint_count, _ = search(log_path, "interrupt")
    # For B, additionally prove the partial text is dropped & unsearchable
    b_partial_search = None
    if scenario.get("partial"):
        pc, _ = search(log_path, scenario["partial"])
        b_partial_search = pc
    # Surface 3: API
    session = make_session(str(td), log_path, backend)
    api_t = api_tail(session)
    api_t_events = public_role_summary(api_t.get("events", []))
    api_s_ctrl = api_search(session, user_text)
    api_s_intr = api_search(session, "interrupt")

    verdict_unit = classify(unit_events, scenario)
    verdict_tail = classify(tail_events, scenario)
    verdict_api = classify(api_t_events, scenario)

    return {
        "scenario": name,
        "backend": backend,
        "input_rows": scenario["rows"],
        "unit_extract_events": unit_events,
        "unit_tail_events": tail_events,
        "search_user_prompt": {"query": user_text, "match_count": ctrl_count, "matches": ctrl_matches},
        "search_interrupt_phrase": {"query": "interrupt", "match_count": hint_count},
        "search_partial_text": None if b_partial_search is None else {"query": scenario["partial"], "match_count": b_partial_search},
        "api_tail_events": api_t_events,
        "api_search_user_prompt": {"match_count": api_s_ctrl.get("match_count"), "matches": public_role_summary(api_s_ctrl.get("matches", []))},
        "api_search_interrupt_phrase": {"match_count": api_s_intr.get("match_count")},
        "verdicts": {
            "unit_extract": verdict_unit,
            "unit_tail_page": verdict_tail,
            "api_tail": verdict_api,
            # search can only surface an outcome that the normalizer produced:
            "search": "DEFECT (no interruption row produced -> nothing to find)"
            if "DEFECT" in verdict_unit else "PASS (interruption row searchable)",
        },
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for scenario in SCENARIOS:
            results.append(run_scenario(scenario, td_path))

    raw = {
        "head": _git_head(),
        "no_response_text_reference": _NO_RESPONSE_TEXT,
        "results": results,
    }
    (ARTIFACT_DIR / "proof-output.json").write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")

    # Human-readable summary
    lines: list[str] = []
    lines.append(f"HEAD: {_git_head()}")
    lines.append(f"Generic no-response text (for reference, NOT expected here): {_NO_RESPONSE_TEXT!r}")
    lines.append("")
    for r in results:
        lines.append("=" * 78)
        lines.append(f"SCENARIO {r['scenario']}  (backend={r['backend']})")
        lines.append("input rows:")
        for row in r["input_rows"]:
            lines.append("  " + json.dumps(row))
        lines.append("")
        lines.append("Surface 1 — unit normalization (_extract_positioned_chat_events):")
        lines.append(f"  events = {json.dumps(r['unit_extract_events'])}")
        lines.append(f"  -> {r['verdicts']['unit_extract']}")
        lines.append("Surface 1b — disk tail page (_read_chat_tail_page):")
        lines.append(f"  events = {json.dumps(r['unit_tail_events'])}")
        lines.append(f"  -> {r['verdicts']['unit_tail_page']}")
        lines.append("Surface 2 — transcript search:")
        lines.append(f"  user-prompt control: {r['search_user_prompt']}")
        lines.append(f"  'interrupt' phrase : {r['search_interrupt_phrase']}")
        if r["search_partial_text"] is not None:
            lines.append(f"  partial text       : {r['search_partial_text']}  (>0 would mean partial preserved & searchable)")
        lines.append(f"  -> {r['verdicts']['search']}")
        lines.append("Surface 3 — /api/messages:")
        lines.append(f"  tail events        = {json.dumps(r['api_tail_events'])}")
        lines.append(f"  -> {r['verdicts']['api_tail']}")
        lines.append(f"  search user prompt : match_count={r['api_search_user_prompt']['match_count']}")
        lines.append(f"  search 'interrupt' : match_count={r['api_search_interrupt_phrase']['match_count']}")
        lines.append("")
    summary = "\n".join(lines)
    (ARTIFACT_DIR / "proof-summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[5]), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
