# Clean-room audit: Claude Code context chip parity

Recommendation: **ACCEPT**. I found no blockers. The implementation adds a bounded Claude Code token parser, wires it into the existing rollout token path, preserves the frontend/runtime surface, and the proof artifacts exercise the user-visible chip through a synthetic CC session.

## Blockers

- None.

## Nonblockers / residual risks

- `cc_token_update()` and `_extract_token_update()` return `None` for an unknown-only CC log, which satisfies the clean unknown-model case. Existing last-known-token semantics still mean a hypothetical session that first emits a known-model token and later switches to an unknown model would keep the older chip unless a future clear-token mechanism is added. I do not treat this as blocking because the slice assumes stable session model identity and did not change token-state semantics.
- Browser proof covers the positive chip path. Unknown-model hiding is covered by unit/mechanism checks, not by a separate browser artifact.
- Worktree note: `.memory/tasks/2026-07-07-cc-context-chip-parity/OPS.md` was already unstaged at review start and remained unstaged; no staged files were present.

## File/line evidence

- `codoxear/cc_log.py:61-97` defines conservative model-window mapping: documented 1M prefixes, 200k known families including Sonnet 4.5/Haiku 4.5, and `None` for unmapped models.
- `codoxear/cc_log.py:111-134` parses only CC assistant rows with `message.role == "assistant"`, reads `message.usage` and `message.model`, computes `tokens_in_context = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, excludes `output_tokens`, and emits the existing `_context_token_update` dict shape.
- `codoxear/rollout_tokens.py:6,14-20` imports `cc_token_update()` and checks it after Pi extraction and before the existing Codex `event_msg` token-count branch. The Codex/Pi branch bodies are otherwise unchanged.
- `tests/test_cc_log.py:105-158` adds coverage for conservative mapping, known assistant usage dict shape, cache-token summing with output exclusion, unknown model returning `None`, and rollout extraction returning a CC token.
- Scope check from `git diff --name-status a8059be^..e57f3ca`: product code changes are limited to `codoxear/cc_log.py`, `codoxear/rollout_tokens.py`, and `tests/test_cc_log.py`; the second commit records proof artifacts under `.memory/tasks/.../browser-artifacts/cc-context-chip-19383/`. No frontend, send/queue/attach, runtime busy, or backend launch files changed.

## Proof/test credibility

- `api-sessions-initial.json` shows a synthetic `agent_backend: "cc"` session with model `claude-sonnet-4-5` and token `{context_window: 200000, tokens_in_context: 150000, tokens_remaining: 33616, percent_remaining: 18, reserved_tokens: 16384, max_input_tokens: 183616, as_of: "2026-07-07T04:55:10.000Z"}`.
- `api-messages-tail.json` carries the same token shape and renders the synthetic `user` and `assistant` transcript rows.
- `browser-cc-context-chip.json` shows `#ctxChip` displayed as `flex`, text `Ctx 18%`, and title `Context input: 150000/183616 tokens (16384 reserved; window 200000).` The sessions token, tail token, and browser-observed row token match.
- `fake_cc_context_session.py` makes the fake broker return `token: null`; therefore the non-null API/browser token is derived from the CC log parser, not injected by the broker fixture.
- `docker-calls-compact.json` records `send_count: 0` and `key_count: 0`; the proof only polled state and transcript.
- `VERIFICATION-REPORT.md` records full local, Docker unit, Docker smoke, diff-check, and browser evidence. I reran the focused CC test file and direct artifact/token checks during this audit.

## Commands run

- `git status --short --branch` → branch `recovery/product-gaps`; one pre-existing unstaged task-memory file; no staged files.
- `git show --stat --oneline a8059be e57f3ca` and `git diff --name-status a8059be^..e57f3ca` → confirmed scoped code/test/proof changes.
- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider tests/test_cc_log.py` → `15 passed in 0.44s`.
- `git diff --check a8059be^..HEAD` → no whitespace/check output.
- JSON artifact extraction script → sessions token, tail token, visible chip, transcript roles, and broker call counts matched the report.
- Unknown-model direct check → `cc_token_update_unknown=None` and `extract_unknown_only=None`.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Product changes are limited to codoxear/cc_log.py and codoxear/rollout_tokens.py plus tests/proof artifacts; no frontend, send/queue/attach, runtime busy, or backend launch files changed. cc_log.py:111-134 implements the requested CC assistant usage token parser and rollout_tokens.py:6,14-20 wires it into the existing token extraction path."
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "Unit tests cover mapping, token shape, cache-token summing/output exclusion, unknown model None, and rollout extraction; proof artifacts show /api/sessions token, /messages/tail token, visible #ctxChip, and zero send/key broker calls for a Docker synthetic CC session."
    }
  ],
  "changedFiles": [
    "codoxear/cc_log.py",
    "codoxear/rollout_tokens.py",
    "tests/test_cc_log.py",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/VERIFICATION-REPORT.md",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/browser-cc-context-chip.json",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/api-sessions-initial.json",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/api-messages-tail.json",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/docker-calls-compact.json",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/fake_cc_context_session.py",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/cc-context-chip-driver.js",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/docker-test-19384.txt",
    ".memory/tasks/2026-07-07-cc-context-chip-parity/browser-artifacts/cc-context-chip-19383/docker-smoke-19384.txt"
  ],
  "testsAddedOrUpdated": [
    "tests/test_cc_log.py"
  ],
  "commandsRun": [
    {
      "command": "git status --short --branch",
      "result": "passed",
      "summary": "On recovery/product-gaps; no staged files; one pre-existing unstaged OPS.md task-memory modification."
    },
    {
      "command": "git show --stat --oneline a8059be e57f3ca && git diff --name-status a8059be^..e57f3ca",
      "result": "passed",
      "summary": "Confirmed scoped product changes plus proof artifacts."
    },
    {
      "command": "PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider tests/test_cc_log.py",
      "result": "passed",
      "summary": "15 passed in 0.44s."
    },
    {
      "command": "git diff --check a8059be^..HEAD",
      "result": "passed",
      "summary": "No whitespace/check output."
    },
    {
      "command": "python3 JSON artifact extraction for sessions/tail/browser/calls",
      "result": "passed",
      "summary": "Sessions token, tail token, visible chip text/title, event roles, and zero send/key counts matched the verification report."
    },
    {
      "command": "python3 direct unknown-model token check",
      "result": "passed",
      "summary": "cc_token_update_unknown=None and extract_unknown_only=None."
    }
  ],
  "validationOutput": [
    "Focused audit rerun: tests/test_cc_log.py -> 15 passed in 0.44s.",
    "Proof artifact: /api/sessions and /messages/tail token both equal context_window=200000, tokens_in_context=150000, percent_remaining=18, reserved_tokens=16384, max_input_tokens=183616.",
    "Proof artifact: browser #ctxChip display=flex, text=Ctx 18%, title=Context input: 150000/183616 tokens (16384 reserved; window 200000).",
    "Proof artifact: docker-calls-compact send_count=0, key_count=0.",
    "Recorded verification report: full local suite 1798 passed/128 subtests; Docker test 1797 passed, 1 skipped, 128 subtests; Docker smoke 401 pre-login and 200 post-login."
  ],
  "residualRisks": [
    "Unknown-model hiding has parser/unit evidence and clean unknown-only extraction evidence, but no separate browser artifact.",
    "Existing last-known-token semantics could retain a previous known-model chip if a session later switches to an unmapped model; stable-model sessions are covered."
  ],
  "noStagedFiles": true,
  "diffSummary": "Adds Claude Code assistant usage token parsing and conservative model context mapping, wires CC into rollout token extraction, adds CC token tests, and records Docker/browser proof artifacts.",
  "reviewFindings": [
    "no blockers",
    "nonblocker: unknown-model browser negative path not separately recorded",
    "nonblocker: known-to-unknown model-switch history would keep last known token under existing token-state semantics"
  ],
  "manualNotes": "Review-only audit; repository files were not edited. The only unstaged file observed was .memory/tasks/2026-07-07-cc-context-chip-parity/OPS.md, already present at review start."
}
```
