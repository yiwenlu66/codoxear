## Verification Report — D4 picker disambiguation + Git Workbench truthful state

**Verdict: PASS (both surfaces)** — verification only, no edits/staging/commits.

| | |
|---|---|
| **Repo / commit served** | `/home/yiwen/codex-web-product-recovery` @ `4cf7e3c` ("Render truthful Git workbench state") — clean working tree, untouched |
| **Sandbox** | Docker container `codoxear-verify-19141`, port `127.0.0.1:19141`, image `codoxear-sandbox:latest`, `APP_DIR=/home/tester/.local/share/codoxear`, root `/tmp/codoxear-docker-sandbox-19141` |
| **Isolation** | Preflight OK; host live `~/.local/share/codoxear` and protected `/home/yiwen/codex-web` never touched; fixtures + fake broker (pid 49) lived entirely inside the container |
| **Cleanup** | `agent-browser` session `d4wb-19141` closed; container `codoxear-verify-19141` stopped+removed via sandbox script; other containers untouched; no pkill/killall |

### Surface A — D4 picker disambiguation (commit `98a2072`)

**API** (`/file/search?q=bad` on git repo with raw-byte `bad\xffname.txt` + literal `bad\\xffname.txt`, both tracked): HTTP **200**, no crash. Two candidates with identical display path `bad\\xffname.txt`, one carrying `api_path=codoxear-git-path-bytes-v1:YmFk_25hbWUudHh0` + `non_utf8_path:true` (raw-byte), one literal (`api_path:null`), both tied at score 308 — the new `_entry_identity` key breaks the tie instead of raising `TypeError`. Content round-trip: token → `RAW COLLISION OK` (17 B), literal → `LITERAL COLLISION OK` (21 B). Distinct and correct.

**Browser** (picker search `bad`): two same-display options rendered with visible distinguishing hints in `.fileMenuHint.fileMenuIdentity` **and** the `title` attribute — `current folder · literal name` vs `current folder · non-UTF bytes`. Each opens to its correct distinct content via the token channel. PASS.

### Surface B — Git Workbench truthful state (commit `4cf7e3c`)

**API**: `changed_files` returns tracked changes with numstat, untracked entries flagged `untracked:true`/`state:"untracked"` (add/del null), staged rename `renamed.txt` with `rename:true`/`old_path:"oldname.txt"`, binary `icon.png` explicit (`add/del null`). `git/diff?head=1` returns unified `+/-` diff for text and `Binary files … differ` for binary. Non-repo session git route → **409** with git error text. PASS.

**Browser desktop**: changed-files picker shows stat chips (`+2-1`, `+?-?` binary), rename `oldname.txt → renamed.txt`, `untracked` badges. Modified text `notes.md` in diff mode (Monaco unavailable) renders the **unified diff** (`diff --git … -line one / +line one MODIFIED / +line three added`) read-only with notice `Rich diff unavailable. Showing unified diff (read-only)` — not working-tree content. Non-repo session shows explicit `Not a git repository — no changed files` (`.fileMenuGitStatus`, `role=status`), not a silent empty picker. No stage/commit/checkout controls; Edit labeled unavailable and produces no writable surface (0 save buttons, 0 textareas, 0 contenteditable). PASS.

**Browser mobile 390×844**: diff fallback notice + `+/-` content readable, picker shows 10 candidates with rename + 2 untracked badges, non-repo notice present — **0 px horizontal overflow** throughout. PASS.

### Residual product gaps (not introduced by these commits)

1. **Monaco editor never loads in a clean deployment** (pre-existing D1 packaging gap, documented in prior memory): no `monaco/` vendored, CSP blocks CDN, loader times out at 4 s. This is exactly why the diff/file paths run through the plain-text fallback — which is the path commit `4cf7e3c` improves. Not a regression of the verified commits; the fallback now renders truthful diff content.
2. **Edit button stays enabled** (`disabled=false`) while labeled "Editing is unavailable…". Clicking it opens no writable surface, so it is not a write control — but disabling it would be cleaner. Pre-existing, Monaco-gap-related.
3. **Diff-toggle timing sensitivity**: toggling into diff while a file-view Monaco load (4 s) is in flight, inspected at the exact timeout boundary, can briefly show file-view content before the unified diff commits. Settles correctly within the timeout; not a defect of the rendered result.

Artifacts: `/tmp/codoxear-docker-sandbox-19141/artifacts/A/` (API JSON), `…/artifacts/B/` (API JSON), `…/artifacts/browser/` (screenshots `a01–a04`, `b01–b04`, `m01–m03` + `dom-summary.json`).