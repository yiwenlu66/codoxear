# Unattended Mode — Naming & Semantics Analysis

## 1. Problem in one sentence

The feature internally called "harness" is a **periodic idle-triggered prompt injector** whose user-facing name ("Harness mode") communicates no part of its mechanism or purpose, while the *prompt it actually injects* already self-identifies as "Unattended-mode instructions."

---

## 2. Mechanism (ground truth from code)

The system is a **server-side sweep loop** (`_harness_loop` → `_harness_sweep`, every `HARNESS_SWEEP_SECONDS` ≈ 2.5 s) that, for each session with an enabled config, does the following:

1. **Guard: injection budget** — `remaining_injections > 0` else auto-disable.
2. **Guard: cooldown** — at least `cooldown_minutes` elapsed since last injection *for this session* AND *for this thread/log scope*.
3. **Guard: session idle** — broker reports `busy=False`, both broker queue and server queue empty.
4. **Guard: assistant was last speaker** — tail-scan the log; the most recent chat role must be `assistant`.
5. **Guard: assistant turn aged out** — `now − last_assistant_ts ≥ cooldown_seconds`.
6. **Action** — `self.send(sid, rendered_prompt)` injects the rendered prompt as a user message.
7. **Bookkeeping** — decrement `remaining_injections`; record timestamps per-session and per-scope; if budget hits 0, set `enabled=False`.

### What the prompt says

The injected text (`HARNESS_PROMPT_PREFIX`) opens with:

> **"Unattended-mode instructions (optimize for 8+ hours, minimal turns, minimal repetition, maximal progress)"**

It then prescribes an internal four-section protocol (Deliverables, Completed, Next actions, Parked user decisions), working rules, yield conditions, and an adversarial review gate.

An optional per-session `request` string is appended as "Additional request from user: …".

### State shape

Per session in `harness.json`:
```json
{
  "enabled": bool,
  "request": str,
  "cooldown_minutes": int (≥1),
  "remaining_injections": int (≥0)
}
```

Runtime (not persisted): `_harness_last_injected[sid]`, `_harness_last_injected_scope[scope_key]`.

### Scope deduplication

When multiple sessions share the same `thread_id` (same underlying backend thread), the sweep injects into at most one of them per cooldown window via `_harness_last_injected_scope["thread:<tid>"]`. This prevents double-nudging the same conversation through different Codoxear session views.

---

## 3. Semantic invariant

> **When enabled for a session, the server periodically injects a continuation prompt after the agent has been idle for at least `cooldown_minutes`, up to `remaining_injections` times, then stops.**

This is the complete behavioral contract. The prompt *content* (the long unattended-mode preamble) is an implementation detail of what gets said, not part of the scheduling invariant.

---

## 4. What "harness" means vs. what it does

| Aspect | "Harness" connotation | Actual mechanism |
|---|---|---|
| **Domain** | Testing/CI (test harness); equestrian (horse harness); safety (fall harness) | Autonomous continuation scheduling |
| **Agency** | Implies constraining or controlling the agent | Actually *prompts* the agent to continue working |
| **Relation to user** | None obvious | Acts as user's delegate: injects user messages while user is away |
| **Temporal character** | No implication of timing | Entirely defined by idle detection + cooldown + budget |
| **Self-description** | Name says nothing | The injected prompt already says "Unattended-mode" |

The name "harness" was likely imported from "test harness" thinking during development but describes neither the user mental model nor the mechanism.

---

## 5. Naming recommendation

### User-facing name: **Unattended mode**

Justification:
- The injected prompt already self-identifies with this exact phrase.
- "Unattended" is accurate: the feature's purpose is to keep the agent working when the user is not watching.
- "Mode" is appropriate: it is a per-session toggle with parameters that changes the system's behavior.
- No ambiguity with other features (queue, diagnostics, voice push).

### Specific copy model

| Location | Current | Proposed |
|---|---|---|
| Sidebar badge | `harness` | `unattended` |
| Top-bar button tooltip | "Harness mode" | "Unattended mode" |
| Settings dialog title/checkbox | "Harness mode" | "Unattended mode" |
| Help text heading | "Harness" | "Unattended mode" |
| Help body | "Harness is a per-session idle nudge…" | "Unattended mode keeps a session working while you're away…" |
| `cooldown_minutes` label | "Cooldown time (minutes)" | "Idle cooldown (minutes)" — keep as-is or use "Minutes before next nudge" |
| `remaining_injections` label | "Number of injections" | "Remaining nudges" or "Continuation budget" |
| Badge tooltip | "Harness mode enabled" | "Unattended mode enabled" |
| Error toasts | "harness save error" / "harness load error" | "unattended-mode save error" / "… load error" |
| README § 6 | "Enable Harness mode" | "Enable Unattended mode" |
| Env var docs | `CODEX_WEB_HARNESS_SWEEP_SECONDS` | See § 6 below |

### Rejected alternatives

- **"Autopilot"** — implies the agent chooses its own goals; this feature just re-prompts.
- **"Keep-alive"** — implies TCP/heartbeat semantics; this is application-level continuation.
- **"Auto-nudge"** — too cutesy; doesn't say *why*.
- **"Idle resume"** — confusable with the existing session-resume feature.
- **"Autonomous mode"** — too broad; the agent is always somewhat autonomous.

---

## 6. Internal API / state renaming strategy

### Recommendation: **rename user-facing copy now; alias internal names with a deprecation path**

The feature has the following internal surface area:

#### A. Server JSON API (client-facing)

| Endpoint / field | Current | Action |
|---|---|---|
| `GET/POST /api/sessions/<id>/harness` | path segment `harness` | **Add** `/api/sessions/<id>/unattended` as the primary route. Keep `/harness` as a silent alias (same handler) for backward compat with older cached `app.js`. Remove alias after one release cycle. |
| Response fields `enabled`, `request`, `cooldown_minutes`, `remaining_injections` | generic enough | **Keep as-is**. These are mechanism-descriptive, not tied to the "harness" name. |
| Session list fields `harness_enabled`, `harness_cooldown_minutes`, `harness_remaining_injections` | "harness_" prefixed | **Add** `unattended_enabled`, `unattended_cooldown_minutes`, `unattended_remaining_injections` to the list response. Serve both old and new keys during transition. After one release cycle, drop `harness_*` keys. |

#### B. Persisted state files

| File | Current | Action |
|---|---|---|
| `~/.local/share/codoxear/harness.json` | filename contains "harness" | **Load from both `harness.json` and `unattended.json`**, preferring `unattended.json` if it exists. Write to `unattended.json`. On first write, rename/migrate `harness.json` → `unattended.json`. This is the same atomic-replace pattern already used for all other state files. |

#### C. Environment variables

| Variable | Current | Action |
|---|---|---|
| `CODEX_WEB_HARNESS_SWEEP_SECONDS` | "harness" in name | **Accept both** `CODEX_WEB_UNATTENDED_SWEEP_SECONDS` (preferred) and the old name. Document the new name. Log a warning if only the old name is set, after one release cycle. |
| `CODEX_WEB_HARNESS_MAX_SCAN_BYTES` | same | Same dual-accept strategy. |

#### D. Python internals (server.py)

| Symbol | Current | Action |
|---|---|---|
| `HARNESS_PATH`, `HARNESS_PROMPT_PREFIX`, `HARNESS_DEFAULT_*`, `HARNESS_SWEEP_SECONDS`, `HARNESS_MAX_SCAN_BYTES` | module-level constants | **Rename** to `UNATTENDED_*`. These are not part of any external contract. |
| `_render_harness_prompt`, `_clean_harness_*`, `_harness_loop`, `_harness_sweep` | private functions | **Rename** to `_render_unattended_prompt`, `_clean_unattended_*`, `_unattended_loop`, `_unattended_sweep`. |
| `SessionManager._harness`, `._harness_last_injected`, `._harness_last_injected_scope`, `._load_harness`, `._save_harness`, `.harness_get`, `.harness_set` | instance attrs/methods | **Rename** to `._unattended`, etc. Add one-line `harness_get`/`harness_set` aliases that delegate, until all call-sites are updated. |

#### E. JavaScript / CSS internals (app.js, app.css)

| Symbol | Current | Action |
|---|---|---|
| `harnessBtn`, `harnessMenu`, `harnessMenuOpen`, `harnessCfg`, etc. | JS local variables | **Rename** to `unattendedBtn`, `unattendedMenu`, etc. These are not API; renaming is safe. |
| `.badge.harness`, `.harnessMenu`, `.harnessGrid` | CSS classes | **Rename** to `.badge.unattended`, `.unattendedMenu`, `.unattendedGrid`. |
| `iconSvg("harness")` | icon key | **Change** to `iconSvg("unattended")`. |

#### F. Test files

| File | Action |
|---|---|
| `tests/test_harness_sweep.py` | **Rename** to `tests/test_unattended_sweep.py`; update all internal references. |
| `tests/test_harness_input_source.py` | **Rename** to `tests/test_unattended_input_source.py`; update string assertions to match new JS variable names. |
| Other test files that reference `_harness` / `_save_harness` / `_load_harness` in setup stubs | **Update** references to new names. |

---

## 7. Compatibility strategy summary

| Layer | Old name | New name | Transition |
|---|---|---|---|
| UI copy (button, badge, help) | "Harness mode" | "Unattended mode" | Hard switch (single deploy) |
| API route | `/harness` | `/unattended` | Dual-serve for 1 cycle |
| API response keys | `harness_*` | `unattended_*` | Dual-emit for 1 cycle |
| Persisted JSON file | `harness.json` | `unattended.json` | Read-both, write-new, migrate on first write |
| Env vars | `CODEX_WEB_HARNESS_*` | `CODEX_WEB_UNATTENDED_*` | Accept both, warn on old-only |
| Python internals | `harness*` / `_harness*` | `unattended*` / `_unattended*` | Hard rename (no external contract) |
| JS/CSS internals | `harness*` | `unattended*` | Hard rename (bundled with server) |
| Tests | `test_harness_*` | `test_unattended_*` | Hard rename |

**"1 cycle"** here means: one `develop` → `main` release where both old and new names work, followed by removal of old aliases in the next.

---

## 8. Validation checks

After the rename, these must hold:

1. **Functional invariant**: enabling unattended mode on a session, waiting for idle + cooldown, and observing an injected prompt must still work identically. Validated by the existing `test_harness_sweep.py` tests (renamed).

2. **State migration**: a server started with only `harness.json` on disk must load it, serve correct unattended config via the new API, and on next save produce `unattended.json`. Validated by a new migration test.

3. **API backward compat** (during transition): `GET /api/sessions/<id>/harness` returns the same payload as `GET /api/sessions/<id>/unattended`. `POST` to either path has the same effect. Validated by a new route-alias test.

4. **Session list dual-emit** (during transition): the response contains both `harness_enabled` and `unattended_enabled` with identical values. Validated by asserting key parity in the session-list test.

5. **Env var fallback**: setting only `CODEX_WEB_HARNESS_SWEEP_SECONDS=5` must still change sweep interval. Validated by a new env-var test.

6. **UI copy**: no occurrence of the string "harness" (case-insensitive) remains in user-visible text after the rename. Validated by a grep assertion in `test_unattended_input_source.py`.

7. **CSS/JS rename**: `test_unattended_input_source.py` asserts the new variable names exist and old ones do not.

8. **Full test suite green**: all 357+ tests pass in the Docker sandbox after the rename.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **User confusion during transition**: user sees "unattended" in new UI but "harness" in old cached JS | Low | Single-file JS; hard refresh clears cache. Add cache-busting query param to `app.js` on rename deploy. |
| **State file race**: server crashes between reading `harness.json` and writing `unattended.json` | Low | The atomic-replace pattern (write tmp → `os.replace`) already handles this. Migration is idempotent. |
| **Env var ambiguity**: both old and new env vars set to different values | Low | Document: new var wins. Log a warning if both are set. |
| **grep/search breakage**: developers searching for "harness" find nothing | Low | Leave a one-line comment at the old constant site: `# Formerly "harness mode"; renamed to "unattended mode" in <commit>.` |
| **Over-rename**: renaming the injected prompt prefix string itself | Zero risk | `HARNESS_PROMPT_PREFIX` → `UNATTENDED_PROMPT_PREFIX` is an internal constant rename; the *content* of the string (which already says "Unattended-mode") stays identical. |
| **`_is_scaffold_user_text` does not filter unattended-injected messages in preview** | Pre-existing | The `_first_user_message_preview_from_log` function filters AGENTS.md and environment_context scaffolding but does NOT filter unattended-mode injected prompts. If an unattended nudge is the first non-scaffold user message, it will show as the session preview. This is a pre-existing issue, orthogonal to the rename, but the rename is a good time to add an `_is_unattended_prompt_text` check. |

---

## 10. Discriminating observations

These would change the formulation if their answers differ from assumed:

1. **Is there any external tooling, script, or automation that calls `/api/sessions/<id>/harness` outside the bundled `app.js`?** If yes, the API alias transition period must be longer or permanent. Evidence: grep the user's home directory and any CI configs for `harness` API calls.

2. **Do any third-party forks or integrations depend on the `harness_enabled` key in the session list response?** If yes, same extended transition. Evidence: check GitHub forks or published wrapper scripts.

3. **Does the user want to rename the *prompt content* (the "Unattended-mode instructions" preamble) as part of this change, or keep it fixed?** The current analysis assumes the prompt stays identical and only the *feature name* changes. If the user wants to revise the prompt, that's a separate content-change task.

4. **Is the `request` field name clear enough, or should it become `extra_instructions` / `user_request`?** The current `request` is ambiguous (HTTP request? feature request?). "extra_instructions" would be more mechanism-accurate. This is a lower-priority rename that could ride along or be deferred.

---

## Summary

The feature is a **server-side idle-continuation scheduler** that injects a structured prompt into agent sessions after a configurable quiet period. The name "harness" describes none of this. The prompt itself already uses "Unattended-mode." The rename to **"Unattended mode"** aligns user-facing copy with the injected prompt, eliminates the jargon, and requires no behavioral change. Internal renames are safe because no external contract depends on the Python/JS symbol names. API and state-file transitions use dual-serve/dual-read patterns that are already established in the codebase.
