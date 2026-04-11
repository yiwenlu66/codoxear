# Pi models.json Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Pi `New session` form source provider choices from `~/.pi/agent/models.json`, switch model suggestions when the provider changes, and preserve manually typed model ids.

**Architecture:** Keep Pi config parsing on the server, extend the Pi launch-default payload with a provider-to-model mapping, and let the web dialog derive provider-scoped datalist suggestions from that mapping. Reuse the dialog's existing model-touched tracking so provider changes only replace the model value when the user has not manually edited it.

**Tech Stack:** Python server defaults parsing, pytest-backed unittest coverage, Preact + TypeScript dialog state, Vitest DOM tests.

---

## Files

- Modify: `codoxear/server.py`
- Modify: `tests/test_launch_defaults.py`
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/components/new-session/NewSessionDialog.tsx`
- Modify: `web/src/components/new-session/NewSessionDialog.test.tsx`

## Tasks

### Task 1: Backend defaults and tests

- [x] Add failing tests for provider-scoped `provider_models`.
- [x] Add failing tests for preserving valid configured defaults.
- [x] Add failing tests for skipping auth-only providers in launch choices.
- [x] Add failing tests for skipping empty providers when choosing fallback defaults.
- [x] Implement Pi launch-default parsing from `models.json` only.
- [x] Keep top-level `models` aligned with the selected provider for compatibility.

### Task 2: Frontend provider-linked behavior and tests

- [x] Add `provider_models` to the launch-default frontend type.
- [x] Add failing tests for provider-linked Pi model suggestions.
- [x] Add failing tests for untouched-vs-touched model replacement.
- [x] Add a failing regression test for `codex -> pi -> provider change` auto-updating.
- [x] Implement provider-scoped Pi model choices in `NewSessionDialog`.
- [x] Reset backend-switch touch state so a fresh Pi backend selection still auto-manages its model until the user edits it.

### Task 3: Verification

- [x] Run `python -m pytest tests/test_launch_defaults.py -q`.
- [x] Run `cd web && npx vitest run src/components/new-session/NewSessionDialog.test.tsx`.
- [x] Re-review the diff for spec compliance and code quality.

## Notes

- `Model` remains a freeform input with a datalist.
- The current worktree already had unrelated local changes in overlapping frontend files, so this work was layered on top without reverting or isolating those edits.
- No commit was created because the overlapping files already contained local work outside this feature.
