# Codoxear `develop` Acceptance Summary

Date: 2026-06-12
Branch: `develop`
Base branch protected: `main` was not merged into or modified.

## Acceptance candidate

`develop` is the single integrated acceptance branch for the major refactor/new-features program. The branch keeps Codoxear's current product model: one shared broker architecture for CLI/web sessions, minimal UI, GTD-style sidebar without nesting, and sparse chat rendering.

The implementation avoided whole stale PR-branch merges. Accepted work was reimplemented or integrated as small commits with targeted tests, then validated in the isolated Docker sandbox.

## Integrated work

### Foundation and validation

- Added `scripts/codoxear-docker-sandbox` and `docker/sandbox.Dockerfile` for isolated server/test validation.
- Repaired two pre-existing baseline failures before feature work: stale cwd file-history deletion and voice summary prompt wording.
- Kept runtime tests on isolated app/session state under Docker homes, not the live Codoxear app dir.

### Accepted PR-compatible fixes

- PR #13 static/package polish: URL-prefix-safe logo path, nested logo package data, optional immutable static cache headers, and no backend-tab rebuild while the new-session modal is open.
- PR #14 tooltip fallback for icon buttons.
- PR #19 quiet client disconnect handling.
- PR #17 stale broker sidecar pruning.
- PR #12/#15 Pi log-binding hardening: preserve and prefer declared Pi session logs.

### Product improvements

- Full conversation transcript copy/export with explicit oversized-log rejection.
- Hidden-tab adaptive session polling: visible cadence remains fast; hidden cadence slows.
- Loaded-scope long-chat navigation: previous/next loaded user turn and search over rendered/loaded messages.
- File picker local-first fuzzy results while full-project search is pending or failed.
- New-session recent provider/model reuse through the existing model combobox.
- Adjacent duplicate assistant chat events are deduped within an assistant stretch in server page/live extraction and in the client live-append path, while repeated answers after a new user turn remain visible.

### Unattended mode

- Public Harness surfaces were cleanly renamed to Unattended mode with no `/harness` route, no `harness_*` public fields, no old state filename, and no old env var.
- Internal Python/JS/CSS/test identifiers were later renamed to Unattended terminology as well.
- Remaining `Harness` strings in tests are negative assertions guarding against reintroducing the old public contract.

### Thinking/reasoning capability semantics

- Pi reasoning choices are now model-aware from Pi `models.json` metadata.
- `reasoning:false` maps to `off`; explicit effort lists constrain UI and API validation.
- Unsupported Pi model/effort combinations fail loudly instead of silently downgrading.
- Codex remains constrained to the known Codex effort enum; no per-model Codex claim is made because no authoritative per-model capability source was found from local Codex help/config inspection.

### Claude Code backend

- Added minimal `cc` backend support through the same broker/log/session abstractions used by Codex and Pi.
- Added `codoxear/cc_log.py` for Claude Code log normalization and tests for chat extraction, idle/busy inference, backend registration, session log discovery, launch defaults, and packaged Claude logo assets.
- UI exposes a Claude backend tab and hides provider/Fast controls for Claude.

## PR decisions

| PR | Decision | Final handling |
|---:|---|---|
| #12 | Accept selectively | Declared Pi session-log binding implemented with tests. |
| #13 | Accept selectively | Static/package/cache/modal polish implemented with tests. |
| #14 | Accept selectively | Default tooltip fallback implemented with source test. |
| #15 | Accept selectively | Combined with #12 Pi log-binding hardening. |
| #16 | Reject | Whole Preact/workspace rewrite conflicts with minimal/no-nesting philosophy. |
| #17 | Accept selectively | Missing sidecar pruning implemented with tests. |
| #18 | Defer | Auth/vendor Monaco changes require separate product/security/package-size review. |
| #19 | Accept selectively | Client-disconnect quiet handling implemented with tests. |
| #21 | Mine selectively | Minimal Claude Code backend support implemented; broader interactive prompt UI deferred. |
| #22 | Defer | macOS launch semantics need separate targeted review/testing. |
| #10/#11/local PR-ish branches | Defer/already covered | No whole-branch merge; mine only if future evidence identifies a current gap. |

## Validation evidence

Latest full code-validation evidence after the final runtime-affecting change:

- `scripts/codoxear-docker-sandbox test` → `429 passed, 2 skipped`.
- `python3 -m py_compile codoxear/rollout_log.py` passed for the assistant chat-extraction change.
- Local and Docker `node --check codoxear/static/app.js` passed for the client live-append dedupe change.

Browser validation ran only against isolated Docker servers with isolated/synthetic state:

- Port `18791`: login, topbar controls, New Session Codex/Pi/Claude tabs, and Claude mode hiding provider/Fast while showing reasoning `medium`.
- Port `18792`: synthetic 320-message Codex transcript loaded a recent tail window, found a loaded search marker exactly once, navigated loaded user turns, and loaded older history to the beginning.
- Port `18793`: renamed Unattended menu/API/sweep path opened in browser, saved settings through `/api/sessions/<id>/unattended`, persisted `unattended.json`, and decremented the injection budget after the isolated sweep injected once.

## Negative evidence and scoped limitations

- A first synthetic long-chat fixture without Codex `session_meta` failed discovery loudly. This validated fail-closed log binding for invalid synthetic Codex logs, but the fixture was not used as UI evidence.
- The synthetic long-chat fixture omitted `end_turn:true`, so that browser run is not idle/busy evidence. Existing idle tests cover the valid Codex `end_turn:true` shape.
- Minimal Claude Code support is unit/source/browser-plumbing validated, not proven against a long real Claude Code session.
- No real Codex/Pi/Claude session creation was run because doing so requires explicit authorization for real binaries/credentials in sandbox state.
- No real mobile-device network/performance trace, Monaco/file-viewer race test, zsh/oh-my-zsh startup test, or full real long-transcript performance run has been performed.

## Parked user decisions

1. Accept `develop` as the candidate branch, or request additional sandbox-realistic validation.
2. Authorize real backend binaries/credentials in isolated sandbox state if live-like Codex/Pi/Claude session creation must be tested.
3. Approve any future merge from `develop` to `main`; no merge to `main` is authorized by this task.
