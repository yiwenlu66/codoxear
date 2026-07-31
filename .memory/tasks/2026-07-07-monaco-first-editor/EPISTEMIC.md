# Epistemic model

## Phenomenon
Codoxear's file editor/diff is now Monaco-first in clean deployments. The corrected defect was structural: Monaco assets were absent from the shipped/static route graph, and UI failure paths treated Monaco absence as a supported plain textarea editor or plain unified-diff surface.

## Supported mechanism
- Clean deployment support is supplied by vendored `monaco-editor@0.55.1` AMD `min/vs` assets under `codoxear/static/monaco/vs`, committed license/third-party notices, `/monaco/...` static routing, recursive package-data inclusion, and recursive static asset hashing.
- `index.html` loads `monaco/vs/loader.js` before file editor/app wiring. `createMonacoLoader()` configures AMD `paths.vs` to the same-origin `monaco/vs` path.
- CSP stays within the existing `worker-src 'self' blob:` contract: the old `data:` worker override is gone and Monaco uses vendored same-origin worker assets.
- Editable text files become writable only when Monaco renders a file editor. Monaco failure renders explicit code-editor-unavailable UI and read-only preview; the file edit affordance explains the unavailable editor and click handling blocks edit entry rather than enabling textarea editing.
- Repository diff rendering requires Monaco diff editor from `/git/file_versions` data. The former plain unified-diff fallback fetch/render path is gone.
- Save conflict handling remains tied to the Monaco model: Keep preserves the unsaved Monaco draft; Reload discards it and reloads disk after confirmation.

## Evidence
- Functional implementation: `c0e8979 Require Monaco for file editor and diff`.
- Local validation before commit: JS syntax checks passed; focused Monaco/static pytest passed (`90 passed, 25 subtests`); wheel-content probe confirmed required Monaco files inside the built wheel; full local pytest passed (`1776 passed, 132 subtests`). See OPS.
- Docker/browser proof: `c145158 Record Monaco editor browser proof`; proof hygiene: `01e0029 Normalize Monaco proof headers`. Clean Docker server on port 19317 served `/monaco/vs/loader.js` with 200; browser opened `notes.txt` with `.monaco-editor` and no `.filePlainEditTextarea`; UI save persisted edits; `changed.md` diff rendered `.monaco-diff-editor`; simulated loader absence rendered explicit unavailable/read-only UI with no writable textarea and no plain diff substitute; conflict Keep/Reload behaved correctly.
- Independent review: `fa42594 Record Monaco cleanroom review`. Critic verdict: nonblocker, no Monaco substrate blocker found.

## Ruled out
- Writable `plain-edit` textarea as a fallback editor.
- Plain unified diff as an accepted repository diff substitute.
- CDN/runtime network dependency for Monaco.
- Package-only success without browser execution: Docker/browser proof exercised the routed assets from a clean container.

## Known caveat
`git diff --check d61ffae..HEAD` reports whitespace in vendored Monaco/ThirdPartyNotice files. This is third-party asset hygiene, not a product behavior defect. The proof header artifact was normalized; remaining failures are confined to vendor bytes that should not be rewritten casually after browser proof.

## Current conclusion
Monaco-first editor/diff is accepted. The next product target is upload expansion: server-owned staged attachment list plus multi-file picker, preserving staged-file identity, backend-readable paths, explicit pending state, and visible failure semantics.
