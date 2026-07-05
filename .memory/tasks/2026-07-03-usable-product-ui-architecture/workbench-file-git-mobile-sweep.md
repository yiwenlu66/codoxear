# Workbench file / git / mobile surface sweep

Scope: file workbench + git workbench + mobile readability, exercised in an
isolated Docker sandbox (no live app dir, no backend inference). Findings ranked
by user impact. No product files were edited (review-only sweep); no
staging/commits.

Sandbox: `codoxear-sandbox-19130`, `HOME=/home/tester`, port 19130 (not 8743).
Isolation preflight OK. Two fake broker sessions were registered (real Unix
control socket answering `{"cmd":"state"}`, valid sidecar JSON, `agent_backend=pi`,
no backend process): `wb-repo` (cwd = a throwaway git repo) and `wb-nonrepo`
(cwd = `/home/tester`, not a git repo). The repo fixtures include: tracked text
(`notes.md`, `src/app.py`), modified/deleted/untracked/staged-mixed states, a
real 1×1 PNG, a real binary blob, a >2 MB binary, an accented-unicode filename,
a genuinely non-UTF-8 filename (raw byte `0xff`, surrogate-encoded), and a
nested git repo.

Screenshots are in
`.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/`
(`wb01..wb05`).

## Backend (API) behavior — verified correct

These passed cleanly and need no action:

- `GET /api/sessions/<id>/file/read` — text/markdown/image/pdf/video/download_only
  kinds; 404 missing; traversal `../../etc/passwd` → 404.
- `GET .../file/search` (git-repo cwd) — uses `git ls-files`, scores + truncation
  correct.
- `GET .../file/list` — correct for UTF-8-only trees.
- `POST .../file/write` — save with version ✓; stale-version conflict → 409
  `{conflict:true, version:<current>}` ✓; `create:true` new file ✓;
  create-existing → 409 ✓; path traversal `../escape.txt` → 400
  `"path escapes session cwd"` ✓.
- `GET .../git/changed_files` — merged unstaged+staged, numstat add/del, binary
  shown as `+? -?`; HTTP 200.
- `GET .../git/diff` — unified diff; binary file → `"Binary files ... differ"`.
- `GET .../git/file_versions` — `current_exists`/`base_exists`; deleted file →
  `current_exists:false, base_exists:true`.
- Non-repo session git routes → 409 with the git error text; read-only, no write
  implied.

Path-boundary enforcement (symlink/escape/repo-root) is sound for both the file
and git surfaces.

## Defects

### D1 (P1, blocker for the editor/diff surface) — Monaco editor never loads in a clean deployment; Edit permanently disabled, diff degraded

Evidence (browser, desktop + mobile):
- Opening any text file shows `Plain text fallback / monaco loader timed out.
  Showing a read-only plain-text view.` and the **Edit file** button is
  permanently `disabled`.
- Opening a git-changed file and toggling diff shows
  `notes.md - diff / Plain text fallback / Rich diff unavailable: monaco loader
  timed out. Showing a read-only plain-text view.` and renders the **current
  working-tree content**, not a diff (screenshot `wb02-monaco-fallback-diff.png`).

Mechanism (high confidence, isolated):
- `codoxear/static/app_file_editor.js` `createMonacoLoader` waits up to 4 s for
  `window.require` (the AMD loader from `monaco/vs/loader.js`), then fails.
- Nothing provisions that loader:
  - No `<script>` tag for any monaco loader in `index.html`
    (`grep -ic monaco` on the served page = 0).
  - `codoxear/static/monaco/` does not exist in the repo and has never been in
    git history; `GET /static/monaco/vs/loader.js` → 404.
  - `docker/sandbox.Dockerfile` does not vendor or fetch Monaco; there is no
    `package.json`/build step anywhere.
- A CDN fallback is impossible by design: CSP is `script-src 'self' 'unsafe-inline'`
  and `test_static_assets.py` explicitly asserts `assertNotIn('src="https://', index)`.
- Tests only assert the literal `const base = resolveAppUrl("monaco/vs");` exists
  in the editor source (`tests/test_static_assets.py:206`); no test asserts the
  loader is served or that the directory exists — so the defect is invisible to
  the suite (the known "pytest green ≠ usable" failure mode).

Impact: in any clean checkout / Docker deployment the entire file-editor
capability is unavailable and the diff view is a misleading plain-text fallback.
The viewer, markdown preview, image/PDF/video/download paths, and the git
workbench do not depend on Monaco and remain functional.

Suggested fix direction (not applied): vendor `monaco-editor` under
`codoxear/static/monaco/` and load `monaco/vs/loader.js` via a `'self'`-compliant
`<script>` (or add a provisioning step), plus a static-asset test that the loader
is actually served. This is a packaging/provisioning gap, not an isolated code
patch, so it was not edited in this review-only sweep.

### D2 (P2, impairing) — `file/list` and walk-mode `file/search` return HTTP 500 on non-UTF-8 filenames; raw Python error reaches the user

Evidence (API + browser):
- `GET /api/sessions/wb-repo/file/list` → **500**
  `{"error": "'utf-8' codec can't encode character '\udcff' in position 65: surrogates not allowed"}`
  (the repo contains a file whose name has raw byte `0xff`).
- `GET /api/sessions/wb-nonrepo/file/search?q=name` → **500**, same encoder
  error (walk mode hits the surrogate-named file under the cwd tree).
- Browser file-picker for the non-repo session, query `name`: the menu renders
  the raw internal error text to the user —
  `'utf-8' codec can't encode character '\udcff' in position 109: surrogates not allowed`
  + a `Retry` button (screenshot `wb03-filelist-surrogate-500.png`).

Mechanism (high confidence, isolated):
- `list_session_relative_files` (`client_file_paths.py`) and
  `search_walk_relative_files` (`file_search.py`) walk with the OS default
  `surrogateescape` and return the raw surrogate strings as JSON path values;
  the response is JSON-encoded to UTF-8, which rejects lone surrogates → 500.
- The git workbench already solves this exact case: `git_ops.path_json_text`
  (`backslashreplace`) + `git_path_token`/`api_path` keep non-UTF-8 names
  round-trippable; `git ls-files` runs with `errors="replace"`. `file/list` and
  walk-mode `file/search` do not use that path-safe serialization.
- Note: git-mode `file/search` on the same repo returns a lossy but non-crashing
  `"\"bad\\377name.bin\""`; only the walk/list paths crash.

Impact: any session whose cwd tree contains a non-UTF-8 filename loses the plain
file list and (for non-repo cwds) file search entirely, with an internal error
exposed to the user. Boundary scenario the product explicitly supports elsewhere
(git paths), so it is a real consistency gap, not an exotic case.

Suggested fix direction (not applied): run the walk-mode/list path values
through the same surrogate-safe serialization used by the git path layer
(`path_json_text`, and expose a token when surrogate bytes are present), and map
the remainder to an explicit user-facing error instead of a 500.

### D3 (P3, polish / mobile ergonomics) — File-viewer toolbar touch targets are 34×34 px (<44 px minimum)

Evidence (mobile 390×844): the five `.fileViewer .icon-btn` controls
(Toggle diff / Toggle markdown preview / Edit file / Download file / Close) and
the picker option rows are each ~33–34 px tall/full-width. They fit the viewport
(no horizontal overflow) and content is readable (screenshots `wb04-mobile-file-viewer.png`,
`wb05-mobile-picker-changed.png`), but the icon buttons sit below the 44×44 px
(iOS) / 48 dp (Android) touch-target minimum. The picker rows are full-width so
they remain comfortably tappable; the small icon buttons are the concern.

Mechanism: CSS sizing of `.fileViewer .icon-btn` (34 px), not a layout bug.
Low-confidence-on-severity (real but polish-tier); no functional break.

## What was verified clean (no defect)

- File picker: opens, lists git-changed files with `+N -M` stats, filters live
  on query, offers "Create new file: <q>", opens files by click.
- Open-by-path resolution, recent/mentioned/changed sections, session switch.
- Markdown preview toggle (renders), image preview (`<img>` natural size correct),
  download-only message ("not renderable as text, markdown, image, or PDF. Use
  Download instead.").
- Git changed_files / diff / file_versions / binary diff / deleted-file /
  nested-repo / non-repo-409 boundary — all correct and read-only.
- Path-traversal / symlink / outside-repo guards on read, write, and git paths.
- Mobile 390×844: no horizontal overflow, picker and changed-files list
  readable, status text legible.

## Residual notes / boundaries

- The fake sessions carry synthetic 2024-07-02 timestamps, so the sidebar shows
  "732d ago"; that is the fixture, not a product bug.
- `file/list` for the non-repo cwd returns `[]` (empty) even when `file/search`
  finds files via walk — a minor inconsistency possibly worth a follow-up, but
  the UI picker is driven by `file/search`, so it is not user-visible in normal
  flows. It is downstream of D2 (the same endpoint crashes once a non-UTF-8 name
  is present), so fixing D2's serialization should subsume it.
- No live runtime was touched; host `~/.local/share/codoxear` untouched; no
  staging or commits.
