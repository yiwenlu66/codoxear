# CC Unknown-Model Token Clearing — Clean-Room Adversarial Review

## VERDICT: ACCEPT

No blockers. The implementation correctly distinguishes three token-observation states (none/update/clear), routes the clear signal through all four token-state mutation paths, and exposes only `dict|null` at every public API surface.

---

## Evidence inspected

### Implementation diff (commit e201ac9)

**token_signal.py** (new, 44 lines): Introduces `TokenObservation` dataclass with `kind ∈ {none, update, clear}`. `observed` is True for update/clear, False for none. `public_token` returns the dict only for update, None for both none and clear. `coerce_token_observation` handles legacy dict/None callers. `TokenObservation` is **not JSON-serializable** — any accidental leak to `json_response` would crash the server with `TypeError`, providing an immediate hard fail instead of silent data leakage.

**cc_log.py**: `cc_token_observation()` replaces `cc_token_update()` as the primary extraction function. Returns `TOKEN_NONE` for non-assistant or no-usage rows (correct: no observation to act on). Returns `TOKEN_CLEAR` when context_window is None (unmapped model) or usage ints are unparseable. Returns `token_update_observation(dict)` for valid known-model data. `cc_token_update()` preserved as thin wrapper (`.public_token`) for backward compatibility.

**rollout_tokens.py**: `_extract_token_observation()` scans newest-to-oldest. For CC: checks `cc_observation.observed` — if True (update OR clear), returns immediately without scanning older rows. This is the key fix: a CC clear signal stops the scan instead of falling through to resurrect stale known-model pressure. Pi path unchanged (wraps non-None dict in `token_update_observation`). `_find_latest_token_observation()` and `_find_latest_token_update()` delegate correctly.

**message_routes.py** (handle_messages_live): Uses `_extract_token_observation(objs)` instead of old `_extract_token_update`. For clear: `token_update.observed=True` → `s2.token = token_update.public_token` = None. Passes `TokenObservation` to `message_runtime_snapshot` → `select_runtime_token` → `coerce_token_observation` → returns None. JSON response gets `"token": null`.

**session_log_runtime.py** (update_meta_counters): Incremental scan accumulates `latest_token_observation` via `coerce_token_observation`. Fallback to `_find_latest_token_update` guarded by `not latest_token_observation.observed AND session.token is None` — a clear in the incremental scan prevents fallback. Final write: clear → `current.token = None`.

**session_runtime.py** (select_runtime_token): `coerce_token_observation(token_update)` handles mixed input types. Clear → `observed=True` → returns `public_token` = None. Legacy dict/None callers coerced correctly.

**rollout_idle.py** (_analyze_log_chunk): Uses `_extract_token_observation`. Return type 5th element changed from `dict|None` to `Any`. All callers either coerce or ignore.

**server.py / server_route_deps.py**: Type annotations updated. `_message_runtime_snapshot` accepts `Any`, downstream `ServerRouteDepsFactory.message_runtime_snapshot` accepts `TokenObservation | dict | None`.

### Token-state path audit (all four mutation paths)

| Path | Mechanism | Clear behavior | Verified |
|------|-----------|----------------|----------|
| `handle_messages_live` | `_extract_token_observation` → `s2.token = public_token` | Sets `session.token = None` | Test + browser proof |
| `update_meta_counters` | `coerce_token_observation(chunk)` → `current.token = public_token` | Sets `session.token = None` | Test |
| `select_runtime_token` (runtime snapshot) | `coerce_token_observation` → return `public_token` | Returns `None` for API | Test |
| `_find_latest_token_update` (fallback) | `_find_latest_token_observation → .public_token` | Returns `None` | Test (cc_log finder test) |

### Sentinel leak audit

- `session.token` is only ever set to `dict` or `None` (via `public_token`). Never set to a `TokenObservation` object.
- All API JSON responses use `token_val` from `select_runtime_token` (returns `dict|None`) or direct `session.token` (always `dict|None`).
- `/api/sessions` reads `facts.token` from session listing, which reads `s.token`.
- `TokenObservation` fails `json.dumps()` with `TypeError` — verified at runtime.
- **No sentinel leaks possible.**

### Pi/Codex regression check

- `pi_log.py`: No changes in the diff. `pi_token_update()` still returns `dict|None`.
- In `_extract_token_observation`, Pi tokens are wrapped via `token_update_observation(pi_token)` — same behavior as before.
- Full test suite (1812 tests, 134 subtests) passed with zero failures.

### Test coverage

| Test | File | What it proves |
|------|------|----------------|
| `test_extract_token_update_known_then_unknown_cc_usage_clears` | test_cc_log.py | Batch: known→unknown returns None |
| `test_extract_token_update_unknown_only_cc_usage_has_no_public_token` | test_cc_log.py | Batch: unknown-only returns None |
| `test_extract_token_update_assistant_without_usage_preserves_prior_cc_token` | test_cc_log.py | Batch: no-usage preserves older known token |
| `test_latest_token_finder_stops_at_unknown_cc_usage` | test_cc_log.py | File scan: clear stops at unknown, returns None |
| `test_messages_live_unknown_cc_usage_clears_stale_session_token` | test_message_routes.py | Live route: TOKEN_CLEAR reaches snapshot, session.token=None, body.token=None |
| `test_update_meta_counters_clears_existing_token_on_newer_unknown_cc_usage` | test_sessions_pending_log_idle.py | Session runtime: existing token=512 cleared to None, no fallback invoked |
| `test_message_snapshot_clear_signal_beats_stored_session_token` | test_sessions_pending_log_idle.py | Runtime snapshot: TOKEN_CLEAR overrides stored session token |

### Proof artifacts

- **Before unknown row**: `/api/sessions` token `{tokens_in_context: 4500, context_window: 200000, percent_remaining: 98}`. Browser `#ctxChip`: visible, enabled, "Ctx 98%".
- **After unknown row**: `/api/sessions` token `null`. `/messages/tail` token `null`. Browser `#ctxChip`: `display:none`, `disabled:true`, empty text. Transcript preserved both KNOWN_MODEL_TOKEN_SENTINEL and UNKNOWN_MODEL_CLEAR_SENTINEL texts.
- No secrets in committed artifacts (marker check passed, only `preferred_auth_method="apikey"` as known static false positive).

---

## Non-blocking concerns

1. **Loose type annotations**: `server.py:_message_runtime_snapshot` uses `Any` for `token_update` parameter instead of `TokenObservation | dict | None`. `rollout_idle.py:_analyze_log_chunk` uses `Any` in 5th tuple position instead of `TokenObservation`. Functionally correct (all callers coerce or ignore), but imprecise for static analysis.

2. **`token_update_observation(None)` edge case**: If called with `None`, produces `TokenObservation("update", None)` where `observed=True` and `public_token=None`. Unreachable in current code (only called with dicts from `_context_token_update`/`pi_context_token_update`), but the factory function doesn't validate its input.

Neither concern is a product risk.

---

## Acceptance criteria check

| Criterion | Status |
|-----------|--------|
| Unknown-model CC usage clears stale known-model token pressure | ✅ Met |
| No-usage CC assistant rows preserve existing token state | ✅ Met |
| Known-model token math remains input+cache_read+cache_creation (output excluded) | ✅ Met (no formula changes) |
| Public API exposes dict-or-null only, no internal sentinel | ✅ Met |
| Product proof covers /api/sessions, /messages/tail, browser #ctxChip | ✅ Met |
| Codex/Pi behavior unregressed | ✅ Met (1812 tests passed, pi_log unchanged) |
