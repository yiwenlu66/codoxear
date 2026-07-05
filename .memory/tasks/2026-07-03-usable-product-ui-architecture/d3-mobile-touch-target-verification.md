# D3 mobile file-viewer touch-target verification

Functional commit: `363d232 Raise mobile file viewer touch targets`.

## Boundary

- Verified from detached worktree `/tmp/codoxear-d3-verify-9e07db0` checked out at `363d232`, so concurrent non-UTF openability edits in the main worktree were not part of this evidence.
- Docker browser sandbox: port `19135`, name `codoxear-d3-browser-19135`, root `/tmp/codoxear-d3-browser-19135`.
- The standard sandbox script preflight passed. The script's build step could not reach Docker Hub metadata, so the container was started manually from the already-cached `codoxear-sandbox:latest` image with the same isolation shape: throwaway `/home/tester`, no host app dir, readonly repo mount, non-live port.
- Container-only fake session `d3-touch` used cwd `/home/tester/work/touchrepo` and a normal text file `notes.txt`.
- Cleanup: exact named container removed and named browser session closed.

## Browser observation

At viewport 390x844, the browser opened `#session=d3-touch`, opened File Workbench via `View file`, and measured `.fileViewer .icon-btn:not(.fileTouchBtn)`.

Visible toolbar buttons:

| Button | Width | Height | Min width | Min height |
|---|---:|---:|---:|---:|
| Toggle diff | 44 | 44 | 44px | 44px |
| Edit file | 44 | 44 | 44px | 44px |
| Download file | 44 | 44 | 44px | 44px |
| Close | 44 | 44 | 44px | 44px |

`allVisibleAtLeast44=true`; `horizontalOverflow=false`; file viewer rect was `x=0,w=390,right=390`.

Screenshot: `browser-artifacts/d3-mobile-fileviewer-touch-targets.png`.

## Negative evidence absorbed

The first browser measurement against the unamended D3 patch found `Edit file` at `38x44`: the broad `.fileViewer .icon-btn:not(.fileTouchBtn)` mobile rule lost to the earlier ID-specific `#fileEditBtn { min-width: 38px; }`. The committed fix adds `.fileViewer #fileEditBtn { min-width: 44px; }` inside the mobile block and a source test that checks this exact cascade seam.

## Validation

- Targeted local: `python3 -m pytest -q tests/test_mobile_toast_source.py tests/test_static_assets.py tests/test_mobile_zoom_accessibility_source.py tests/test_file_viewer_source.py` -> `65 passed, 25 subtests`.
- Manual Docker smoke equivalent: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`.
