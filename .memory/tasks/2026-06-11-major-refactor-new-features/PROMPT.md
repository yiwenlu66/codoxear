## Objective
Recover Codoxear product gaps on `/home/yiwen/codex-web-product-recovery`, branch `recovery/product-gaps`, until the branch is a validated, reviewable candidate for explicit user approval.

Completion requires user-facing workflows to work end-to-end under scoped evidence, not merely tests or implementation scaffolding. The acceptance branch is `recovery/product-gaps`; `main` remains live-safe and must not be merged or modified without explicit user approval.

Prioritize reliability, UX/accessibility, bounded refactors that preserve invariants, feature polish, and safe live/backend validation where the evidence justifies it. Broad structural/frontend refactoring may proceed only after product gaps are fixed or explicitly scoped.

## Workbench
1. Implement failed-launch in-chat recovery panel. See: Send/queue recovery promise; Backend/capability promise; Sparse UI promise.
2. Validate failed-launch recovery panel. See: Validation criteria; Evidence criteria.
3. Investigate and repair video preview/transcoding. See: File/context promise; Video preview/transcoding bug.
4. Investigate and repair Pi busy-after-interrupt state. See: Backend/capability promise; Responsiveness promise; Pi interrupt/busy bug.
5. Pick the next responsiveness or UX/refactor tranche from `recon/refactor-entry-checkpoint.md`. See: Responsiveness promise; Refactor promise.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery`.

Protected live checkout: `/home/yiwen/codex-web` on `main`.

Active branch: `recovery/product-gaps`.

Task memory: `.memory/tasks/2026-06-11-major-refactor-new-features/`.

Current checkpoint artifact: `recon/refactor-entry-checkpoint.md`.

Project architecture and development guide: `AGENTS.md`.

Codoxear is a Linux-first companion UI for local CLI agent sessions. Supported backends are Codex, Pi, and scoped Claude Code (`cc`) work.

Server state for real sessions lives under `~/.local/share/codoxear`; tests and browser validation should use isolated app/session state instead.

Recent recovery work established or hardened send/queue recovery, transcript search/loading, Pi provider pass-through, sidecar metadata validation, file picker match highlighting, git helper extraction, Details copy, and Details → New like this launch-copy semantics.

User-reported active markdown issues from 2026-06-15: chat markdown code blocks render with an undesirable dark style; markdown tables should not overflow chat width and should wrap or stay contained.

User-reported active backend/media issues from 2026-06-13: ffmpeg video transcoding/preview does not work and may never have worked; Pi sessions can stay busy after interruption.

Credential and provider context for integrated testing: use the user's existing local backend/provider configuration under `~/.pi/agent` and installed CLI backends when safe; never print, copy, or commit credential values.

Preferred cost-efficient model/provider for general integrated workloads: `deepseek-v4-flash`.

Preferred Claude-specific workload: `occ-claude` with `claude-haiku-4-5`.

Use headless agent-browser for UI testing when useful.

User wants long unattended progress on features, refactoring, and deep UI/UX experience, with minimal turns and no repetition.

Parked limits include incomplete Codex/Claude live-response evidence, real mobile-device and assistive-tech evidence, slow-network and huge-transcript evidence, smooth Jump to latest, non-UTF-8 Git filename byte-literal behavior, and atomic symlink containment against concurrent local mutation.

## Task specifications
Product acceptance is organized by user promises and invariants, not by implementation workstreams or test-count progress.

Sparse UI promise: the main chat top bar exposes only identity/state and immediate chat actions; utilities belong in contextual surfaces, not a generic `More` dumping ground.

New-session promise: users can choose coherent backend/provider/model/reasoning settings; Pi provider names come from Pi CLI/config authority; synthetic diagnostics `provider_choice` must not become actual Pi provider state; providerless Pi sessions remain providerless through copied presets, recent selections, memory, parsing, and start requests.

Send/queue recovery promise: HTTP `/send` success is a commit boundary; unknown commit state blocks unsafe mutation paths; recovery UI may explain, review, and clear explicit markers but must not silently resume unsafe work.

Long-session orientation promise: users can search, navigate, and regain position in large sessions without dense log-style chat clutter; loaded/all-transcript count evidence must not be discarded during navigation refresh.

File/context promise: file picker, viewer, refs, Git paths, and fuzzy search preserve literal identity while giving clear empty/loading/error/mobile states.

Markdown rendering promise: fenced code blocks should be readable and visually consistent with Codoxear's minimal light UI; markdown tables should stay within chat width by wrapping/containing cell content, including on mobile-sized layouts.

Video preview/transcoding bug: reproduce the user-reported ffmpeg preview failure with representative fixtures, identify the exact failing mechanism, and repair without hiding conversion errors behind silent fallback.

Pi interrupt/busy bug: reproduce or fixture Pi busy state after interruption, identify whether the mechanism is Pi log normalization, broker/sessiond state, interrupt route semantics, or UI state clearing, and repair with scoped evidence.

Responsiveness promise: polling, transcript loading, rendering, and network behavior avoid stale confusion or avoidable busywork under realistic isolated browser/session conditions.

Backend/capability promise: Codex, Pi, Claude Code, provider, model, service tier, and reasoning controls say only what evidence supports; unsupported or unknown combinations fail loudly or remain explicitly scoped.

Refactor promise: extract or restructure only where the controlling invariant is understood and validation can show behavior preservation; do not substitute architecture motion for missing product behavior.

Validation criteria for each implemented tranche: focused tests for changed logic, browser-level evidence for changed UI when feasible, full local `python3 -m pytest -q`, Docker `scripts/codoxear-docker-sandbox test`, clean git diff review, and atomic commits with scoped messages.

Evidence criteria for claims: distinguish observation from interpretation, preserve negative evidence and reviewer counterexamples, state what the validation does not prove, and keep unsupported backend/mobile/accessibility claims parked.

## Constraints
Do not edit `/home/yiwen/codex-web`.

Do not merge or promote to `main` without explicit user approval.

Do not touch live sessions.

Do not touch, stop, restart, or kill the live server.

Do not kill `codoxear-broker` or underlying backend CLI processes.

Use isolated Docker/browser/runtime state for validation whenever possible.

Do not print secrets, credentials, tokens, private logs, or provider configuration values.

Do not commit runtime artifacts, sockets, live app state, bulky scratch data, or secrets.

Do not use `git add -A`, `git add .`, or broad staging.

Do not use a generic `More` menu as a dumping ground for unrelated UI actions.

Do not add smooth `Jump to latest` behavior until scheduler/runtime harness evidence exists.

Do not add silent fallbacks that hide broken contracts or unsupported combinations.

Do not claim live backend, mobile-device, assistive-technology, slow-network, or huge-transcript coverage without direct evidence.
