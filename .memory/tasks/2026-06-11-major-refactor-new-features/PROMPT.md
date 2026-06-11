## Objective
Create and execute a major refactoring/new-features program for Codoxear with one acceptance target: a single `develop` branch containing the integrated candidate work. Workstreams are interacting and need not be forced into independent final branches. The agent should determine the branch topology that best preserves evidence and reviewability, but the user should only need to evaluate one `develop` branch for acceptance. Nothing may merge to `main` until the user explicitly approves.

Done means:
- A single `develop` branch exists as the acceptance candidate and contains the integrated, reviewable result.
- Temporary topic branches/worktrees may be used for exploration, PR cherry-picks, or risky changes, but they are implementation scaffolding, not the final deliverable.
- Commits on `develop` are atomic enough to review and ordered by dependency/mechanism where possible.
- Each accepted change preserves Codoxear's design philosophy: CLI and web share the same broker; UI remains minimal; sidebar remains GTD-style without nesting; chat view deliberately omits low-value details.
- The integrated result may include creative product improvements beyond the enumerated workstreams when evidence or product judgment shows they make Codoxear better without violating the hard constraints or product philosophy.
- All server/browser validation runs against a standalone Docker test instance with isolated app/session state, not live sessions or the live server.
- The user receives evidence for what changed, what was tested, what remains uncertain, which PRs were accepted/rejected/deferred, and why.

## Workbench
Current status: `develop` is the integrated acceptance candidate; `main` remains untouched. The branch contains the selected PR-compatible fixes, isolated Docker validation sandbox, responsiveness and long-chat navigation changes, clean public Unattended-mode rename, internal Unattended naming cleanup, Pi model-aware thinking-effort constraints, minimal shared-broker Claude Code support, file picker/search ergonomics, recent provider/model reuse, and deterministic pressure-test regressions.

Current evidence base:
1. Full isolated Docker suite is green after the latest code changes: `425 passed, 2 skipped`.
2. JS parse checks passed for `codoxear/static/app.js` in local and Docker contexts.
3. Browser validation ran only against isolated Docker servers and synthetic state: Claude/new-session controls on port 18791, long-chat loaded search/navigation/history on port 18792, and renamed Unattended menu/API/sweep behavior on port 18793.
4. Recon and plan artifacts are preserved in `recon/`; task observations and validation are in `OPS.md` / `EPISTEMIC.md`.
5. Implementation/source files no longer contain Harness terminology; remaining Harness strings in tests are negative assertions guarding against old public compatibility.

Selected next tasks, if the user asks to continue beyond this acceptance candidate:
1. Run live-like backend session creation tests for Codex/Pi/Claude in isolated sandbox state, if the user authorizes use of required binaries/credentials and confirms no live sessions should be touched.
2. Add deeper browser/device pressure tests for mobile network/performance, Monaco/file-viewer races, and full long-transcript interaction if the user wants more acceptance evidence.
3. Pressure-test shell startup variants such as zsh/oh-my-zsh in an isolated environment if those environments are available.
4. Otherwise, await the user's acceptance decision for `develop`; do not merge to `main` without explicit approval.

Observed failures / negative evidence:
- An initial Python 3.11 sandbox image could not collect tests using newer f-string syntax; switching the sandbox to Python 3.13 fixed the measurement artifact.
- Two pre-existing baseline failures were fixed before feature work: stale cwd file-history deletion and voice summary prompt wording.
- A first synthetic long-chat fixture omitted Codex `session_meta`; discovery failed loudly as designed, and the fixture was corrected.
- Synthetic long-chat rows omitted `end_turn:true`, so that browser run is not idle-status evidence; existing idle tests cover the valid `end_turn:true` shape.
- In the Unattended browser smoke, remaining injections dropped from 3 to 2 because the isolated sweep immediately injected once into the idle synthetic session; this was expected for that fixture and validates the renamed sweep path.

Open blockers / unknowns:
- No user authorization yet to use real backend credentials/binaries for live-like Codex/Pi/Claude sandbox session creation.
- Claude Code support is minimal and test/browser-plumbing validated, not proven against a long real Claude session.
- Codex model-specific reasoning capability remains less constrained than Pi because no current authoritative per-model Codex capability source was established.
- Real mobile-device performance, Monaco/file-viewer browser races, zsh/oh-my-zsh startup behavior, and full real long-transcript performance remain untested.

## Context
Required project context:
- `AGENTS.md` for architecture, design philosophy, development reminders, and safe restart constraints.
- `codoxear.server`, `codoxear.broker`, `codoxear.sessiond`, `codoxear.rollout_log`, `codoxear.pi_log`, and `codoxear/static/*`.
- Runtime app directory convention: `~/.local/share/codoxear` for real sessions; do not use it for this task's server/browser testing.
- Existing git history and open GitHub PRs for regression signals and candidate changes.
- Provider configs in `~/.pi/agent` may inform test workloads.
- Unattended-mode code paths in `codoxear/server.py` and `codoxear/static/app.js`; these schedule/send prompts after an idle assistant turn with cooldown and remaining-injection limits.
- Current chat view rendering, message normalization, scroll behavior, session message APIs, and any browser performance costs that affect long-conversation navigation.

Testing preferences from user:
- Prefer `deepseek-v4-flash` for cost-efficient workloads.
- Prefer `occ-claude`'s `claude-haiku-4-5` for Claude-specific workloads.
- Use a headless agent browser when useful for UI testing.

Design philosophy that must constrain PR acceptance and new implementation:
- CLI and web share the same broker model.
- Minimal UI.
- GTD-style sidebar without nesting.
- Deliberate omission of details in chat view.
- Navigation affordances for long conversations should help users regain orientation without making the chat view visually dense or exposing low-value implementation detail.
- Prefer replacing semantically wrong subsystems over layering patches onto confused structures.
- Define semantic invariants before implementing queueing/streaming, chat navigation, or UI state machinery.

## Working style
- Be creative and product-minded. The numbered workstreams are not a ceiling.
- Do whatever makes Codoxear materially better when the intervention respects the hard operational constraints, product philosophy, and validation requirements.
- Prefer coherent product improvements over narrow checklist execution, but preserve evidence: state the mechanism, expected user benefit, validation path, and any tradeoff.
- Do not use open-ended latitude as permission for unchecked scope creep, silent fallbacks, live-runtime risk, or UI complexity that contradicts the project philosophy.

## Branch topology
- Final acceptance target: `develop`.
- `main` is not an acceptance target and must not receive merges without explicit user approval.
- Workstreams are not fully orthogonal. The agent should choose the branch topology after inspecting dependencies, conflicts, and PR shapes.
- Allowed scaffolding: temporary topic branches, throwaway worktrees, or local experiment branches when they reduce risk or preserve evidence.
- Required integration behavior: accepted work is integrated into `develop`; rejected/deferred experiments are documented and not left as required deliverables.
- Keep reviewability by using atomic commits, explicit commit messages, and evidence notes rather than by pretending every workstream can be a separate final branch.

## Task specifications
The workstreams below are interacting areas of investigation and implementation, not an exhaustive checklist or branch map. They should inform each other, and branch topology should follow the evidence rather than the numbered list. Additional product improvements are allowed when they are causally motivated, validated, and compatible with the design philosophy.

1. Architecture review and refactoring
   - Review current server/broker/log/UI architecture and identify mechanisms that are semantically confused or overly patched.
   - Refactor only where a clearer invariant is identified and validation can show preservation of behavior.
   - Avoid broad rewrites without evidence that they reduce complexity or fix a real mechanism.

2. Review and cherry-pick GitHub PRs
   - Inventory open PRs, summarize the behavior each PR changes, and decide accept/reject/defer.
   - Accept only PRs compatible with the design philosophy above.
   - If accepted, merge or cherry-pick into `develop` or a temporary integration branch that will be folded into `develop`; do not merge to `main`.
   - Preserve rationale for rejected PRs, especially when rejection is due to UI complexity, broker divergence, sidebar nesting, or chat detail creep.

3. Optimizations: UI responsiveness and network traffic
   - Measure current polling/network behavior and responsiveness, especially under slow mobile-network conditions.
   - Optimize only after identifying the dominant mechanism, such as redundant polling, oversized payloads, inefficient DOM updates, or unnecessary re-rendering.
   - Validate with browser/network evidence from the standalone Docker instance.

4. Claude Code support
   - Add Claude Code (`cc`) support analogous to existing `codex` and `pi` backends.
   - Preserve the shared broker architecture rather than adding a separate path.
   - Define log discovery, metadata, launch defaults, message normalization, idle/busy detection, and session creation semantics before implementation.
   - Use `occ-claude`'s `claude-haiku-4-5` for Claude-specific test workload if an actual backend test is needed and safe.

5. New-session view ergonomics
   - Consider combining provider and model into a single `provider/model` selector.
   - Improve recent-list ergonomics so the user does not need to type the full `provider/model` name.
   - Keep the UI minimal; avoid a complex nested picker unless evidence shows it is necessary.
   - Validate keyboard and mobile ergonomics in headless/browser tests where possible.

6. File viewer combobox and fuzzy search
   - Re-check combobox logic.
   - Ensure recently viewed files are easy to select.
   - Ensure project files can be fuzzy-searched.
   - Preserve simple file-viewer/editor ergonomics and test edge cases around selection, typing, recent entries, and missing files.

7. Pressure-test frequent git-history bugs
   - Mine git history for frequently mentioned bugs around file viewer/editor ergonomics, rollout log binding, and startup error handling.
   - Reproduce or falsify each bug in the standalone Docker instance.
   - Add regression checks or harden behavior where the mechanism is understood.
   - Record bugs that cannot be reproduced with the evidence and conditions tested.

8. Overall UI cleanliness and responsiveness
   - Improve visual cleanliness and interaction responsiveness without adding conceptual complexity.
   - Keep chat view intentionally sparse and sidebar GTD-style without nesting.
   - Validate desktop and mobile-ish layouts in the Docker/headless-browser environment.

9. Rename/recast `harness mode`
   - Treat "harness mode" as inaccurate terminology unless investigation proves otherwise.
   - Determine the actual mechanism and choose a more accurate name. Current observations indicate the feature is server-side idle-triggered prompt injection for unattended continuation.
   - Prefer a user-facing name that describes the behavior, e.g. "Unattended mode" or another mechanism-accurate term. Do not keep vague "Harness mode" copy.
   - Rename the public API/state surface cleanly; do not preserve `/harness` or `harness_*` compatibility aliases. Internal implementation names may be cleaned separately only if the diff remains reviewable.
   - Validate the renamed feature in the standalone Docker instance.

10. Long-conversation chat navigation ergonomics
   - Improve navigation when conversations become long.
   - Consider lightweight search within visible/loaded chat messages, jump to previous/next user message, jump to latest/oldest relevant turn, and time-based navigation or markers.
   - Preserve deliberate chat-detail omission: navigation can use timestamps/roles as affordances, but should not turn the main chat view into a dense log/debug transcript.
   - Optimize for mobile and keyboard ergonomics; controls should be discoverable without occupying excessive space.
   - Validate against synthetic or fixture long conversations in the standalone Docker/browser environment, including slow-device or slow-network conditions where feasible.

11. Thinking-level / reasoning-effort capability semantics
   - Treat Codex thinking-level support as incomplete until the actual launch/session semantics are inspected and validated.
   - Pi may not support every thinking effort for every provider/model combination; capability constraints must be backend- and model-aware.
   - Do not present thinking levels as universally valid if the backend/model cannot honor them. Prefer explicit capability-aware options, disabled/annotated choices, or backend-scoped defaults.
   - Do not add silent downgrades from unsupported thinking efforts to some other value. Unsupported combinations should fail loudly, be blocked before launch, or be explained in UI/API response semantics.
   - Validate representative supported and unsupported Codex/Pi thinking-effort combinations in the isolated Docker environment where feasible, using current provider/model config and upstream docs when needed.

Cross-workstream verification criteria:
- `python3 -m pip install -e .` succeeds in the isolated test environment where applicable.
- Server starts in Docker with test-only app/session state and required password configuration.
- API and UI checks exercise relevant changed behavior without touching live sessions or the live server.
- Changed Python paths have targeted tests or equivalent scripted validation.
- Changed UI behavior has browser-level evidence when feasible.
- Long-conversation navigation changes are validated against large enough conversations to expose scroll/search/performance and orientation problems.
- Thinking-level/provider/model changes are validated for both supported and unsupported combinations, and evidence shows whether the selected backend actually honors the requested effort.
- Git status is reviewed before and after each workstream/integration step; only intended files are changed.
- `develop` is the branch presented for user acceptance, with any temporary branches clearly identified as non-final scaffolding.

## Constraints
Hard rules:
- Do not touch live sessions.
- Do not touch, stop, restart, or kill the live server.
- Do not kill `codoxear-broker` or underlying backend CLI processes.
- Test everything in a standalone Docker instance with isolated app/session state.
- Present only one final acceptance branch: `develop`.
- Determine branch topology from actual dependencies and evidence; do not force the numbered workstreams into separate final branches.
- Do not merge anything to `main` without explicit user approval.
- Do not commit secrets, provider credentials, live logs, runtime sockets, app state, or bulky scratch artifacts.
- Do not use `git add -A`, `git add .`, or broad staging when unrelated files may exist.
- Preserve existing design philosophy: shared broker for CLI/web, minimal UI, GTD-style sidebar without nesting, deliberate chat-detail omission.
- Creative/product latitude does not override the hard ops constraints, `develop` acceptance target, validation requirements, or product philosophy.
- Do not add silent fallbacks that hide broken contracts; prefer explicit errors or explicit degraded-mode semantics.
- If a PR or implementation conflicts with the design philosophy, reject or redesign it rather than accepting the conflict.
