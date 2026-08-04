# OPS — issue triage evidence trail (append-only, UTC timestamps)

## 2026-07-31T09:12Z — deploy baseline

- Repo `/home/yiwen/codoxear` on `main`. `git fetch origin main`: local already at origin/main HEAD `1976c30b` ("Merge recovery/product-gaps into main"). No new commits to pull.
- pipx venv `/home/yiwen/.local/share/pipx/venvs/codoxear`: `direct_url.json` had `dir_info: {}` (NON-editable install). Site-packages copy dated Jul 31; confirmed no `__editable__` finder / no codox `.pth`. ⇒ a git pull alone does NOT redeploy.
- Reinstalled: `pipx install --force /home/yiwen/codoxear` → codoxear 0.1.0, py3.14.6, apps broker/server/sessiond.
- Restarted service: `systemctl --user daemon-reload && systemctl --user restart codoxear-server.service`.
- Health: `systemctl --user status` → active (running), Main PID 113745, override adds tailscale serve ExecStartPost (`:8443` → `127.0.0.1:8743`). Live pi-broker tmux session `codoxear` survived restart (expected).
- Route probes: `GET /` → HTTP 200; `GET /api/sessions` → HTTP 401 (auth gate). Version import `0.1.0`. Deploy verified against HEAD `1976c30b`.

### Prediction (logged before action)
Predicted `/` 200 + `/api/sessions` 401 after reinstall+restart because the service had just been reinstalled from the same HEAD it was already running; restart should be a no-op on behavior. Observed exactly that. No anomaly.

## 2026-07-31T09:55Z — ISSUE-4 stale living-doc fix

- Scope scan: stale path/branch literals found in living `.memory/project/` (ARCHITECTURE.md:3-4, PRODUCT_GAP_STATUS.md:5) and in 105 historical task records under `.memory/tasks/`.
- Decision: rewrite ONLY living project docs. Historical task records left intact — OPS.md is append-only audit; those records truthfully describe the recovery-worktree era. Rewriting them would falsify history.
- Edits:
  - ARCHITECTURE.md:3-4 → single repo `/home/yiwen/codoxear` on `main`; recovery merged (`1976c30b`); removed two-checkout model; added non-editable pipx redeploy note.
  - PRODUCT_GAP_STATUS.md:5 → re-scoped to `main`; flagged scout verdict as SUPERSEDED by ISSUE-2/ISSUE-3; pointer to live ISSUES.md.
  - issue-triage EPISTEMIC.md anomaly cleared.
- Verify: grep `.memory/project/` for lines asserting old paths as CURRENT active checkout → none. Old-name mentions remaining are explicit "obsolete/removed" explanations (false positives).
- No code/deploy change (memory docs are agent-local, not served).

## 2026-08-04T11:04Z — ledger reconciliation

- Audited `.memory/project/PRODUCT_GAP_STATUS.md` and the issue-triage `ISSUES.md`, `OPS.md`, `PROMPT.md`, and `EPISTEMIC.md` against reachable git history.
- Reconciled historical done items to commits: ISSUE-1 `39c7ac28`; ISSUE-2/3 `ddae3e62`; ISSUE-4 `1c047e9b`; ISSUE-5 `51a4c098`; ISSUE-6 `83fe9dab`; ISSUE-7 `9ac87675`; ISSUE-8 `a7569c3a`; PR #21 `f4a06a2e`; PR #22 `d86bdda6`.
- Reconciled later in-conversation done concerns to their concrete commits in PRODUCT_GAP_STATUS.md, including steering/queued-turn typing continuity `f83645b3`/`2e5602d5`.
- Classified no item as in-progress. Existing uncommitted repository work was deliberately not assigned an owner because the ledgers provide no evidence that it belongs to a recorded issue.
- Kept the only open entries bounded to existing residuals: Pi retry-status compatibility, terminal quiet-window behavior, large-log run-settings replay cost, pre-first-turn slash-command evidence, and real-hardware paper-language judgment. Each now has a discriminating next action in PRODUCT_GAP_STATUS.md.
- Preserved prior closure prose and the append-only OPS history; no code or deployment behavior changed.
