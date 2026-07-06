# Transcript search synthetic no-response fix

Verdict: PASS.

Functional fix: `06930c9 Search synthetic no-response outcomes`.

## Mechanism

`/messages/search` now builds the search stream through the same positioned transcript semantics as tail/history/live: `_dedupe_assistant_chat_events()` followed by `_inject_no_response_events()`. Regular events keep their source record end as `_after_byte`; injected no-response rows use their close record start as `_before_byte` and the close record end as `_after_byte`, so search can attach a `history_cursor` and `load_cursor` that reopen a history window containing the same synthetic row.

This removes the prior second authority in `transcript_search.iter_positioned_chat_events_forward()`, which called `_single_chat_event()` per row and therefore never created synthetic no-response messages.

## Evidence

Local validation:
- Focused route/projection suite: `84 passed`.
- Full local suite: `1725 passed, 132 subtests passed`.
- Original scout probe rerun: both Codex and Claude Code search rows flipped from `DEFECT` to `PASS`; no `DEFECT` lines remained.

Docker/API/browser validation on isolated port `19242`:
- Pre-login `/api/me` returned 401; post-login `/api/sessions` returned 200.
- Fake container sessions were discovered from `/home/tester/.local/share/codoxear`, not host runtime state.
- HTTP `/messages/search` found the synthetic no-response row for `search-codex-noresp` and `search-cc-noresp` with `match_count=1`.
- Each no-response search match included a `load_cursor`; loading `/messages/history` at that cursor returned a window containing the same synthetic no-response row.
- Answered Codex turn returned `match_count=0` for no-response text and `match_count=1` for `CODEX-ANSWER-SEARCH`.
- Claude Code terminal API error search returned real backend text `API Error: 503 Search Proof`, not the generic no-response text.
- Browser UI proof: opening `#session=search-codex-noresp` and `#session=search-cc-noresp`, then searching for the no-response phrase, showed `1/1 loaded · 1 all` and highlighted the rendered no-response row in both sessions.

Docker validation:
- Focused Docker suite: `84 passed`.
- Full Docker suite: `1724 passed, 1 skipped, 132 subtests passed`.

Independent audit:
- Critic `3cfb3527` reported PASS/no blockers. It identified one residual internal resource regression: `count_limit` no longer bounds record consumption because search buffers the bounded window before applying the count limit. The audit judged it non-blocking because result semantics, cursors, truncation flags, ordering, cutoff behavior, malformed tolerance, oversized handling, and dedupe remain correct. If large-log `count_max` performance becomes user-visible, the right fix is a streaming injector state machine that preserves the single no-response authority.

## Artifacts

- `api_search_probe.py`: HTTP route probe.
- `api-search/*.json`: raw HTTP responses for sessions, search, and cursor-loaded history.
- `browser-codex-search.json/png`: browser UI proof for Codex no-response search.
- `browser-cc-search.json/png`: browser UI proof for Claude Code no-response search.
- `fake_search_sessions.py`: deterministic container fake-session generator.
- `container-state.txt`, `server-log-excerpt.txt`, `cleanup.txt`: isolation and teardown evidence.
- `local-full-pytest.out`, `docker-focused-tests.out`, `docker-full-tests.out`, `scout_transcript_outcomes.fixed.out`: validation output.
