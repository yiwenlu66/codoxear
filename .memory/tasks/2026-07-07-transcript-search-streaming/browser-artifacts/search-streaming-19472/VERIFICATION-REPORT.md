# Transcript search streaming verification

## Claim
Transcript search no longer requires whole-log record/event materialization before first-order count limits can stop work, while the user-visible search surface still finds loaded and older transcript matches with correct count/truncation/cursor behavior.

## Functional evidence

- Functional commit: `7fa1fce Stream transcript search events`.
- Local focused validation after integration: `tests/test_transcript_export.py tests/test_message_routes.py` passed (`48 passed`).
- Full local validation after integration: `python3 -m pytest -q` passed (`1827 passed, 134 subtests`).
- Streaming-specific test: `test_count_limited_first_order_search_stops_record_stream` monkeypatches the record iterator and proves `count_limit=5` consumes at most `6` matching records, which fails under the former batch materialization mechanism.
- Existing no-response/search cursor route tests remain in the focused set and passed.

## Docker evidence

- Initial Docker focused run exposed a test portability bug (`unittest.mock` was not imported in Python 3.13 slim). The functional commit was amended to import `mock` explicitly; the retained `docker-focused-19470-initial-fail.txt` records the failure mode.
- Docker focused validation on port `19470` passed `48 passed` after the test portability fix.
- Docker smoke on port `19471` proved the isolated server boundary: pre-login `/api/me=401`, post-login `/api/sessions=200`, app dir `/home/tester/.local/share/codoxear`.

## Browser/API proof

Proof server: exact container `codoxear-search-streaming-19472` on port `19472`.

Container-only fake session:
- Session id: `search-streaming-session`.
- Runtime-only log: `/home/tester/large-search-proof/large-search-session.jsonl`.
- Log size: `539195` bytes, `3002` JSONL rows.
- The log itself was not copied into the repo; only the generator script and reduced summaries were committed.

Direct API evidence:
- `api/search-needle-countmax.pretty.json`: `/messages/search?q=needle&limit=1&text_max=96&count_max=1000` returned `match_count=1000`, `match_count_truncated=true`, first match `bulk needle search row 0000`.
- `api/search-early-target.pretty.json`: `/messages/search?q=EARLY_ONLY_TARGET&limit=1&text_max=96` returned one match with a history/load cursor for `EARLY_ONLY_TARGET first historical match`.

Desktop browser evidence (`browser/search-streaming-proof-result.json`):
- Loaded tail initially contained recent `needle` rows but not `EARLY_ONLY_TARGET`.
- Opening the real chat search UI and searching `needle` showed loaded hits plus `1000+ all`; in-browser API agreed (`apiMatchCount=1000`, `apiTruncated=true`).
- Searching `EARLY_ONLY_TARGET` first showed `0/0 loaded · 1 all`, then clicking the real Next control loaded the older cursor window and rendered `EARLY_ONLY_TARGET first historical match`.
- No horizontal overflow.

Mobile browser evidence (`browser/mobile-search-streaming-proof-result.json`):
- Fresh 390x844 browser session selected the same large session.
- Tail initially contained recent `needle` rows but not `EARLY_ONLY_TARGET`.
- In-browser API agreed on `needle` count truncation (`1000`, `true`).
- The real search UI loaded the older `EARLY_ONLY_TARGET` row through the cursor path.
- No horizontal overflow.

## Boundary

The browser proof demonstrates the user-visible search workflow and large-log route behavior. The specific claim that `count_max` reduces parsing work is proven by the unit-level iterator-consumption test, because external HTTP/browser timing is not a discriminating measurement of iterator consumption.
