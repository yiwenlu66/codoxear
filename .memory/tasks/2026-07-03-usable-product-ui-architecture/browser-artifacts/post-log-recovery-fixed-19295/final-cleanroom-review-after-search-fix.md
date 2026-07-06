I have completed the adversarial review. Here is my final memo.

---

# Final clean-room review — post-log recovery + large-log search fix

**Verdict: ACCEPT. No blockers, no impairing issues.** The prior IMPAIRING gap (I1) is genuinely fixed, load-bearing, and honestly reported. The broader post-log-bound death recovery claim survives scrutiny and is unaffected by the search-only change.

## What I verified (mechanism, not just test counts)

**I1 is really fixed and the fix is load-bearing.** Recovered search for a missing session with an existing `log_path` now runs the *same* whole-log primitive as live search (`_search_chat_log_bounded` → `iter_positioned_chat_events_forward`), which reads the file forward from offset 0 bounded only by a 4 MiB *per-line* cap — not the 2 MiB tail payload. I proved load-bearingness by running the new tests against the pre-fix source in an isolated worktree: old source returns `match_count == 0` for the pre-tail `FIRST_EVENT_SENTINEL`, new source returns `1`. The pre-log fallback test passes on *both* revisions, confirming that path is untouched.

**Attacks on the new search code, all resolved in the code's favor:**
- *Before-cursor semantics:* `before` decoded with `kind="history"` against `_launch_payload_cursor_session` (thread_id + log_path); matches carry `history_cursor` (`_before_byte`) and `load_cursor` (`_after_byte`) with the same identity, so search cursors round-trip with `/messages/history` and `/messages/search?before=`. Verified in `api-proof` (pos=0 head cursor decodes; top-level cursor pos=8349 = truncation boundary).
- *Lifecycle merge:* merged only when `before_byte is None` (latest position), so paging older never re-injects the terminal error. `order=first` appends lifecycle after log matches into remaining slots; `order=latest` concatenates then takes `[-max_matches:]` — lifecycle is the terminal event, so last placement is correct.
- *count_max truncation:* honest. Log search caps at `count_max` and flags truncation; merge subtracts consumed budget, caps lifecycle add, and sets `match_count_truncated` when the cap hides the lifecycle match.
- *Duplication:* none — the lifecycle text is Codoxear-synthesized and absent from the backend log, so the log scan cannot re-find it.
- *Clipping/notification:* applied once on the merged set (`text_max=0` internally, real clip at the end); `_attach_notification_texts` only touches assistant `final_response` and is safe for a missing session.
- *Refactors:* the `raw_events` extractions in `_search_launch_payload_events`, `_launch_payload_events_with_cursors`, and `handle_messages_export` are behavior-identical; the live-session search branch is unchanged.

**Broader recovery claim still holds** (unchanged by a search-only fix, re-confirmed): route id preserved as `session_id`, backend thread id as `thread_id`; backend log read-only with exactly one appended lifecycle error; completed-idle control gets `control_row: null` (no false stopped record); `send`/`enqueue`/`inject_file`/`unattended` all return `404`; export enabled; restart rehydrates from `session_launches.jsonl` (ledger-sourced); tail/history/live/search/export coherent.

**Proof is genuine and honest.** `api-proof-after-final-restart.json`: `large_search_first` = 200 / `match_count:1` / `match_count_truncated:false` / text `FIRST_EVENT_SENTINEL` / usable `load_cursor`; `large_search_stopped` = `match_count:1`; `large_history` loads `FIRST_EVENT_SENTINEL`; `control_row: null`; controls 404. Browser: before = no FIRST_EVENT + visible "Load older"; after = FIRST_EVENT present + `lifecycleErrorCount:1` (no duplication). `VERIFICATION-REPORT.md` states the corrected claim precisely ("Search now searches the real bound log … large recovered logs do not silently miss pre-tail matches"). Independent runs: focused `45 passed`; related suites `119 passed`; LSP clean; Docker logs show `1774 passed, 1 skipped` and smoke `401`/`200`.

## Findings (all NON-BLOCKING)

1. **Stale epistemic claim (memory, not code).** `EPISTEMIC.md:371` still says "tail/live/**search** use a recent tail window, not offset 0." After this fix that is false for `search` — it reads the whole log. The parent's `db6d756` OPS entry already defers the EPISTEMIC update to this verdict. Recommended correction: split the invariant — *tail/live use the tail window; search reads the whole bound log from offset 0 (per-line-bounded); history pages the whole log*.
2. **`_before_byte` leaks into search-match JSON** (visible in `large_search_first.matches[0]`). Pre-existing and identical in the live-session search path — not introduced here; clients ignore unknown keys. Cosmetic.
3. **`launch_ledger.py` `has_older=False` exception branch is untested.** It is reachable (`_extract_positioned_chat_events` has no inner try/except, so a malformed tail record raises). The fix is correct; the pre-fix effect was only a cosmetic "Load older present but no-op" glitch, not data loss.
4. **`order=latest` recovered-search merge and `count_max`+lifecycle interaction are not directly unit-tested.** Logic traced correct; low risk given the symmetry of the merge.

## Residual boundaries (honest, by design)
- Recovered search reads the whole log (4 MiB per-line cap) — same resource profile as the already-accepted live search (the `count_max`-buffering residual applies equally).
- Export still returns explicit `413` for logs > 50 MiB while search/history reach further — loud, not a silent I1-class miss.
- End-to-end browser "type query → jump to a pre-tail match" for the large recovered log is proven by composition (API search returns a usable cursor + browser "Load older" loads FIRST_EVENT), not one combined artifact — the same standard as the accepted recovery tranche.
- Deterministic synthetic logs / stale sidecars in Docker prove Codoxear lifecycle/projection/search semantics, not live provider inference health.

## Notes for the parent
- HEAD advanced from `02aa8ac` to `db6d756` during this review; `db6d756` is memory-only (OPS.md +12 lines) and consistent with my findings. The functional commit under review (`92947c3`) and its proof (`02aa8ac`) are unchanged.
- I made no edits, no staging, no commits. I created and removed one throwaway git worktree at `/tmp/precheck-prefix` (parent commit) purely to prove load-bearingness; it is gone and the main tree is clean. The protected `/home/yiwen/codex-web` checkout was not touched.