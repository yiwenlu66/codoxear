# EPISTEMIC

Append-only scientific ledger for observations, mechanisms, predictions, uncertainty, and scoped claims during the structural refactor + UX review task. Each record must include `date '+%Y-%m-%d %H:%M'`.

## 2026-06-12 10:07
- Working hypothesis: safest first progress is behavior-preserving extraction with explicit compatibility shims and focused tests. High-risk semantic changes, especially busy/idle authority, should remain research-first until current behavior is pinned.

## 2026-06-12 10:07
- Observation: baseline branch validates before source refactoring. Future failures after this point should be interpreted as introduced by the refactor unless traced to an independent measurement artifact.

## 2026-06-12 10:10
- Observation: message cursor code was a narrow cluster inside `server.py` but only depended on HMAC bytes, base64/json helpers, and a minimal session shape. This made it a low-risk first extraction.
- Scoped claim: cursor semantics are behavior-preserved for the tested route/cursor/export paths. The server monolith is reduced without changing public cursor token shape or route JSON fields.

## 2026-06-12 10:13
- Observation: cookie auth logic was another narrow server helper cluster with route-wide impact but small direct dependencies. The extraction uses explicit settings/secret injection, preserving the existing HMAC token shape and refresh-cookie behavior.

## 2026-06-12 10:16
- Observation: Unattended persistence was a self-contained JSON state format inside `SessionManager`. Moving it into a store preserves current behavior while creating the first explicit persistent-state boundary.
- Scoped claim: Unattended state file format, validation errors, sweep behavior, and public no-Harness contract are preserved under focused tests. This extraction does not yet solve the broader multi-store lifecycle problem.

## 2026-06-12 10:17
- Evidence update: the first structural extraction checkpoint preserved the full tested behavior under Docker. This supports continuing with additional mechanical seams rather than stopping at small helper extraction.

## 2026-06-12 10:19
- Observation: GET and POST duplicated URL-prefix redirect/404 logic. Extracting a shared parser is a behavior-preserving route seam with low semantic risk.
- Scoped claim: static and URL-prefix route behavior is preserved under focused tests. The route ladder remains large; this is only the first routing decomposition boundary.

## 2026-06-12 10:23
- Observation: voice/notification/audio routes formed a cohesive route group that did not need direct interleaving with session routes. Extracting them reduces the route ladder while preserving the same handler-level dependencies.

## 2026-06-12 10:25
- Observation: broker had an older JSONL offset reader that would parse partial appended lines and advance offsets differently from the canonical utility. Delegating to the utility reduces parser divergence while explicitly preserving broker missing-file semantics.
- Negative evidence: first wrapper attempt caught `FileNotFoundError` after the utility logged it; adding a pre-existence check preserves the previous quiet behavior.

## 2026-06-12 10:25
- Evidence update: route-prefix/static/voice grouping and broker JSONL dedupe preserve the full suite. This supports proceeding to frontend modularization or further store extraction under the same mechanical-refactor discipline.

## 2026-06-12 10:28
- Observation: backend/defaults helpers were already pure frontend logic but lived as loose functions in the global app closure. A buildless factory creates an explicit frontend module boundary without changing delivery or adding framework/tooling risk.
- Scoped claim: backend config UI semantics are preserved under source/VM checks. This is only the first frontend boundary; most `renderApp()` state remains monolithic.

## 2026-06-12 10:28
- Evidence update: first buildless frontend factory preserved the full tested behavior. Remaining frontend monolith risk is not eliminated, but the refactor path is validated for one pure boundary.

## 2026-06-12 10:36
- Observation: the isolated browser-review server failed at login, before any UX claim could be made. Mechanism: auth extraction moved HMAC helpers to `auth.py`, but the password compare wrapper remained in `server.py` and still needed the module import.
- Scoped claim: targeted auth tests now constrain this wrapper-level regression. Full browser review must use a restarted sandbox to load the fixed module.

## 2026-06-12 11:03
- Observation: tests alone did not reveal user-facing overlay problems. Real browser snapshots showed background controls remained accessible while custom modals were open and transient popovers could stack behind modals.
- Mechanism: custom modals were displayed as ordinary root siblings without making the main `.app` inert/hidden, and modal openers did not close transient overlays such as Unattended or the mobile sidebar.
- Intervention: define a shared modal isolation boundary and close transient overlays before opening custom/native modals. Hide the entire provider field rather than only its control for providerless backends.
- Post-fix evidence: desktop and mobile browser snapshots after the patch contained only modal controls, `.app` was inert/aria-hidden while modals were open, Unattended no longer stacked behind Settings, and Claude/Pi no longer showed an orphan Provider label.
- Scoped claim: overlay accessibility/order and providerless new-session layout are improved for the tested desktop/mobile sandbox flows. This does not prove all nested file-editor subdialogs under real editing races, but source coverage includes the shared subdialog list and browser evidence covers the main modal families.

## 2026-06-12 11:06
- Observation: the only full-suite failure after the overlay fix was a brittle source adjacency assertion, not a behavior failure. Updating it preserved the underlying invariant: pending file opens are still cancelled before the viewer is shown, with modal isolation prep now explicitly in the sequence.
- Scoped claim: the branch currently satisfies the deterministic suite under isolated Docker plus JS parse. Remaining uncertainty is live backend/device behavior, not covered by this sandbox.

## 2026-06-12 11:08
- Correction from user: changing the frontend in the original checkout was not isolated from the user's live server. My prior assumption that Docker sandboxing alone was sufficient was false because the live server may serve static assets from the current repository checkout.
- Revised invariant: all remaining structural refactor and UX work must happen in a separate git worktree; the original `/home/yiwen/codex-web` checkout must remain on the user-safe branch unless the user explicitly asks to change it.

## 2026-06-12 11:32
- Observation: prior summaries treated useful partial work as an acceptance candidate. User review exposed that several nontrivial feature requests were only touched or overclaimed, especially provider/model selection and UI cleanliness.
- Revised commitment: do not use structural refactor as a substitute for missing product behavior. Fix real gaps first; resume refactor only after the feature task is product-complete or explicitly scoped by the user.

## 2026-06-12 11:40
- Revised model: implementation mechanisms and green tests are insufficient acceptance objects. The live claims must be product promises about workflows under invariants, supported by scoped evidence.
- Prediction for recovery: if this ontology is enforced, provider/model and top-bar/action-placement work will be treated as central contract failures, not polish or optional refinements.
