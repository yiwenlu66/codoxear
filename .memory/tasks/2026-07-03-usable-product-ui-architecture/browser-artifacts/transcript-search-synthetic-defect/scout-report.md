Evidence complete. Working tree is clean, no edits/staging performed. Here is the scout finding.

## Verdict: **DEFECT** (narrow, single surface)

The surface-map invariant — *"Search, older-history pagination, and live polling preserve the same transcript messages"* — is **violated by transcript search** for the two *synthetic* outcome rows. History, tail, live, and fresh-process re-read all preserve them.

### Direct observations (route-level, real handlers + signed cursors)

Probe: `/tmp/scout_transcript_outcomes.py` (exercises `handle_messages_search` / `handle_messages_tail` / `handle_messages_history` against fake logs via `MessageRouteDeps`, HMAC secret wired through real `encode/decode_message_cursor`).

| Surface | Codex synthetic no-response | CC synthetic no-response (`system/turn_duration`) | CC terminal `system/api_error` (real text) |
|---|---|---|---|
| `/messages/tail` | PASS (existing tests + mechanism) | **PASS** — row present | **PASS** — `API Error: 503 …` |
| `/messages/history` (older page w/ close in window) | **PASS** — `no-response in history=True`, `pages_walked=1` | (same mechanism; `_read_chat_page_reverse` calls the injector) | PASS (real row) |
| `/messages/live` (split close) | PASS (`test_messages_live_cc_split_turn_duration_emits_no_response`) | PASS (same test) | PASS |
| Fresh process, same disk log | PASS (routes re-read disk via same extractors) | **PASS** — reproduced | PASS |
| **`/messages/search`** | **DEFECT — `match_count=0`** | **DEFECT — `match_count=0`** | **PASS** — `match_count=1` |
| Search control (ordinary user prompt) | PASS (`trigger silent backend` found) | — | — |

So the defect is specific to **search + synthetic rows**: querying `q=backend completed this turn without producing a response` (or any substring of the no-response text) returns zero matches and therefore no `load_cursor`/`history_cursor` for that row. CC terminal `api_error` text is searchable because it is produced by `_single_chat_event` directly from the log row — it is not synthetic.

### Root mechanism

`codoxear/transcript_search.py::iter_positioned_chat_events_forward` builds the searchable event stream by calling `_rollout_log._single_chat_event(record.obj, …)` per record. It **never calls `_inject_no_response_events`**. The synthetic no-response text (`_NO_RESPONSE_TEXT = "The backend completed this turn without producing a response."`) is generated *only* inside `_extract_positioned_chat_events` → `_inject_no_response_events` (`codoxear/rollout_chat_events.py`), which tail/history/live all use, but search does not.

Consequence: the no-response row does not exist in the event stream `search_chat_log_bounded` filters, so:
1. `match_count` / `matches` exclude it for both Codex (`event_msg/task_complete` w/ no assistant) and CC (`system/turn_duration` w/ no assistant).
2. No `history_cursor`/`load_cursor` is attached for the row (cursors are only built for emitted matches in `_attach_search_load_cursors`).
3. The browser's "all-count" search hint undercounts outcome rows that the rendered transcript actually shows — the user sees a row they cannot navigate to or locate via search.

This is exactly the surface-map concern: a transcript message that is visible in tail/history/live is **not** preserved by search.

### Minimal fix target

Route search through the same no-response-injecting extractor as the other readers, preserving byte offsets (search depends on `_before_byte`/`_after_byte` for signed cursors).

- **File:** `codoxear/transcript_search.py` — `iter_positioned_chat_events_forward` (or `search_chat_log_bounded`) must emit the synthetic events. Because search scans forward from byte 0 (bounded by `before_byte`), the open-turn state is built naturally forward; run `_inject_no_response_events` incrementally over the forward record stream with `prior_user_byte=None` at the start (no reverse prior scan needed for a from-0 scan). Each injected event already carries `_before_byte = close_byte` from `_inject_no_response_events`, so `_attach_search_load_cursors` will produce the correct signed `history_cursor`/`load_cursor` for free once the event is in the stream.
- **Constraint to preserve:** assistant dedupe (`_chat_assistant_dedupe_key`) currently runs inside `iter_positioned_chat_events_forward`; the injector must run on the deduped, positioned stream exactly as `_extract_positioned_chat_events` does, so answered turns still suppress injection and the cursor bytes stay consistent with tail/history.
- **Tests:** add to `tests/test_transcript_export.py` (canonical search test file): assert `search_chat_log` / `handle_messages_search` returns `match_count >= 1` with `_NO_RESPONSE_TEXT` and a decodable `history_cursor` for (a) a Codex `user_message` + `task_complete`-no-assistant turn, (b) a CC `user` + `system/turn_duration` turn; and a regression that an answered turn still yields no no-response match.

### Residual notes for the parent

- History/tail/live/restart are **already correct** for synthetic rows — do not widen scope there.
- This defect is disjoint from the in-flight critic `46181ecc` tranche (transcript projection correctness) — it is a search-path-only gap.
- Next ranked uncovered surface after this fix: the log-only stale-`interrupted_idle` browser proof (unit-covered, no dedicated Docker/browser harness for the exact log-advances-while-broker-still-interrupted variant) — lower payoff than this user-facing search gap.

```
acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Read-only scout only. No edits, no staging, no commits. Scope held to the four asked questions; produced evidence-backed DEFECT localization for search + synthetic outcome rows and confirmed PASS for history/tail/live/restart."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git rev-parse HEAD && git status --short",
      "result": "passed",
      "summary": "HEAD=415e46f; working tree clean, nothing staged."
    },
    {
      "command": "python3 /tmp/scout_transcript_outcomes.py",
      "result": "passed",
      "summary": "Route-level probe: search returns match_count=0 for Codex and CC synthetic no-response text; tail/history/live/re-read all preserve the synthetic row; CC terminal api_error real text is searchable (control user-prompt search works)."
    },
    {
      "command": "python3 /tmp/scout_debug_history2.py",
      "result": "passed",
      "summary": "Boundary check: when the silent turn's close lands in an older history window, _read_chat_history_page emits the no-response row (has_no_response=True); when it lands in the tail window, tail emits it. Confirms history path is correct."
    },
    {
      "command": "grep -n 'iter_positioned_chat_events_forward/_inject_no_response/_extract_positioned_chat_events' across codoxear/transcript_search.py codex-web/rollout_log.py codoxear/rollout_chat_events.py",
      "result": "passed",
      "summary": "Source mechanism confirmed: transcript_search.iter_positioned_chat_events_forward calls _single_chat_event only and never _inject_no_response_events; the injector runs only inside _extract_positioned_chat_events (used by tail/history/live)."
    }
  ],
  "validationOutput": [
    "DEFECT: /messages/search cannot find synthetic no-response text for Codex (event_msg/task_complete, no assistant) or Claude Code (system/turn_duration, no assistant). match_count=0, no load/history cursor.",
    "PASS: /messages/search finds CC terminal system/api_error real text (produced by _single_chat_event directly, not synthetic).",
    "PASS: /messages/tail renders synthetic no-response (Codex+CC) and CC terminal api_error.",
    "PASS: /messages/history preserves synthetic no-response when the close row is in the paginated window (route walk + direct _read_chat_history_page).",
    "PASS: /messages/live split-close emits no-response (existing test_messages_live_cc_split_turn_duration_emits_no_response + _prior_open_turn_context path).",
    "PASS: fresh-process re-read from disk reproduces synthetic rows (routes read disk via same extractors; no volatile UI state)."
  ],
  "residualRisks": [
    "Fix must preserve byte offsets (_before_byte/_after_byte) so signed search cursors stay valid; injector must run on the deduped positioned stream to keep suppression semantics aligned with tail/history.",
    "Scout used synthetic fixtures, not a Docker/API harness; mechanism is deterministic from the source path, but the implementing contract should add the Docker/API confirmation."
  ],
  "noStagedFiles": true,
  "diffSummary": "No diff. Read-only scout; no source, test, or memory files were modified, staged, or committed.",
  "reviewFindings": [
    "DEFECT (search-only): codoxear/transcript_search.py::iter_positioned_chat_events_forward does not call _inject_no_response_events, so synthetic no-response rows (Codex event_msg/task_complete and CC system/turn_duration, both with no assistant) are invisible to /messages/search and carry no cursor. Minimal fix: run the same no-response injector over the forward search stream, reusing the injected _before_byte for signed cursors. Tests belong in tests/test_transcript_export.py.",
    "No blockers in history/tail/live/restart paths for synthetic outcome rows."
  ],
  "manualNotes": "Probe artifacts kept under /tmp (scout_transcript_outcomes.py, scout_debug_history.py, scout_debug_history2.py) — outside the repo, not staged. This finding is disjoint from critic 46181ecc's tranche (transcript projection correctness); it is a search-path gap. Next uncovered surface after this fix: the log-only stale-interrupted_idle Docker/browser proof (lower payoff)."
}
```
