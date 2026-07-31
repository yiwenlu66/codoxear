# Interruption transcript-outcome defect — verification report

Verdict: **DEFECT** (proven deterministically at current HEAD).

The product invariant says every sent turn must persistently render one of
`answer / error / no-answer / interruption`. This report proves that a user
turn ending in an interruption/abort renders **only the user row** across every
transcript surface — indistinguishable from an ignored prompt — and, for a Pi
partial-text abort, the already-streamed partial assistant text is discarded.

No fix is implemented. This is a proof of current behavior.

## HEAD / scope

- Repo: `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`.
- HEAD: `55896d5fd253f9bcf98e61127dff71303e128950`.
- `git status --short`: only the untracked artifact directory
  `.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/interruption-outcome-defect/`.
- **No source files or tests were modified.** All artifacts are temporary
  scripts/results under the artifact directory.

## Mechanisms tested

Three minimal synthetic logs, one per scenario, each a user turn followed by an
interruption close:

- **A — Pi empty abort:** `message` user, then `message` assistant `stopReason:"aborted"` with `content:[]`.
- **B — Pi partial-text abort:** `message` user, then `message` assistant `stopReason:"aborted"` with `content:[{text:"I was halfway through"}]`.
- **C — Codex `turn_aborted`:** `event_msg` `user_message`, then `event_msg` `turn_aborted`.

Each scenario was driven through five surfaces:

1. **Unit normalization** — `codoxear.rollout_log._extract_positioned_chat_events` (in-memory records).
2. **Disk tail page** — `codoxear.rollout_log._read_chat_tail_page` (disk `.jsonl`).
3. **Transcript search** — `codoxear.transcript_search.search_chat_log_bounded` (disk).
4. **`/api/messages` tail** — `codoxear.message_routes.handle_messages_tail`.
5. **`/api/messages` search** — `codoxear.message_routes.handle_messages_search`.

Root-cause suppression points (read-only, cited from code, not edited):

- Pi: `PiBackend.chat_event_from_log_row` returns `None` on `pi_assistant_is_aborted_turn(row)` **before** the text check (`codoxear/agent_backend.py:619`), so both empty and partial-text aborts are dropped at extraction. `pi_assistant_is_aborted_turn` is `stopReason == "aborted"` (`codoxear/pi_message.py:94-100`).
- Codex: `CodexBackend.chat_event_from_log_row` has no branch for `event_msg` `turn_aborted` (`codoxear/agent_backend.py:290`), so the close row projects nothing.
- No-response injector deliberately excludes both: `_inject_no_response_events` treats only `event_msg` `task_complete`/`turn_complete` (Codex) and `system` `turn_duration`/`api_error` (CC) as closes (`codoxear/rollout_chat_events.py:294/302`); `turn_aborted` and Pi `message` aborts are neither a chat event nor a close.
- This is pinned as *current intended* behavior by `tests/test_codex_no_response_projection.py::test_pi_aborted_turn_does_not_emit_no_response` (asserts an aborted Pi turn yields events `["user"]` only) and `tests/test_server_chat_flags.py::test_pi_aborted_text_does_not_count_as_last_assistant_chat`.

## Exact commands

```
python3 .memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/interruption-outcome-defect/prove_interruption_defect.py
python3 -m pytest tests/test_codex_no_response_projection.py::test_pi_aborted_turn_does_not_emit_no_response tests/test_server_chat_flags.py -q
```

Raw outputs are preserved in the artifact directory:
`prove_interruption_defect.py` (script), `proof-output.json` (machine-readable),
`proof-summary.txt` (human-readable).

## Observations (per scenario × surface)

For **every** scenario the normalized roles are exactly `["user"]` and search
for any interruption phrase returns `match_count=0`. The control (the ordinary
user prompt is searchable: `match_count=1`) confirms the pipeline is wired
correctly and the silence is specific to the interruption outcome.

| Scenario | Unit `_extract` | Disk tail page | Transcript search (`"interrupt"`) | `/api/messages` tail | `/api/messages` search (`"interrupt"`) |
|---|---|---|---|---|---|
| **A — Pi empty abort** | `["user"]` DEFECT | `["user"]` DEFECT | `match_count=0` DEFECT | `["user"]` DEFECT | `match_count=0` DEFECT |
| **B — Pi partial abort** | `["user"]` DEFECT | `["user"]` DEFECT | `match_count=0` DEFECT | `["user"]` DEFECT | `match_count=0` DEFECT |
| **C — Codex `turn_aborted`** | `["user"]` DEFECT | `["user"]` DEFECT | `match_count=0` DEFECT | `["user"]` DEFECT | `match_count=0` DEFECT |

### Partial-text handling (Scenario B) — explicit

The abort row carried `content:[{type:"text", text:"I was halfway through"}]`.

- Tail/extract events contain **only** the user row — the partial text is **dropped**, not preserved.
- `search_chat_log_bounded(log, "I was halfway through")` → `match_count=0`: the already-streamed partial assistant output is unsearchable too.

So for Pi partial abort the defect is worse than "no outcome row": the partial
answer the user watched stream is actively discarded from the persistent
transcript.

### Asymmetry vs. no-answer

For reference, a Codex turn closing via `task_complete` with no assistant output
*is* projected — the injector synthesizes `"The backend completed this turn
without producing a response."` (text constant from
`codoxear/rollout_chat_events.py:_NO_RESPONSE_TEXT`). The same turn closing via
`turn_aborted` projects nothing. Completed-no-answer gets a persistent row;
interrupted-no-answer gets nothing.

## Conclusion

DEFECT confirmed at `55896d5`. An interrupted user turn leaves no persistent
`interruption` (nor any other) outcome row on any transcript surface — unit
normalization, disk tail, transcript search, `/api/messages` tail, and
`/api/messages` search all return only the user row. Pi partial-text abort
additionally drops the streamed partial assistant text. The behavior is
self-consistent across the layers because all layers funnel through
`_single_chat_event` (Pi abort → `None`) and `_inject_no_response_events`
(Codex `turn_aborted` is neither projected nor treated as a close).

## Boundaries

- **Unit + API-handler proof is complete and deterministic**; no live CLI or
  Docker was used. The API surface was exercised through the real
  `handle_messages_tail` / `handle_messages_search` handlers with a minimal
  fake manager/handler (the same pattern the prior transcript-search scout
  used), so it exercises the same code path a real HTTP request does without
  binding a socket or broker.
- Not exercised here: a full live browser Stop-click (would reproduce the same
  suppression; the scout already traced the frontend-only `setToast("interrupting...")`
  → self-clear in 2.2s with no `app_transcript.js`/`app_message_rows.js`
  interruption row). This is a verification-cost boundary, not an epistemic one
  — the suppression is in the backend normalizer, which the browser render path
  consumes unchanged.
- This is a read-only proof. No fix is proposed or implemented; the correct
  interruption-outcome row (distinct class/text vs. the generic no-response
  text, and Pi partial-text preservation) is a separate product/implementation
  decision.

## Artifacts

- `prove_interruption_defect.py` — the proof script (imports codoxear surfaces; writes no source).
- `proof-output.json` — machine-readable per-scenario × per-surface results.
- `proof-summary.txt` — human-readable summary.
- `VERIFICATION-REPORT.md` — this report.
