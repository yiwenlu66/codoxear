# Destructive confirmation focus safety browser proof

Scope: Docker-only Codoxear instance on port 19492 with a synthetic real session `confirm-focus-session`, a fake broker socket, a pending-attachment compatibility flag, and a real editable file `proof.txt` inside the container workspace.

## Validated behavior

- Destructive `Delete session?` dialogs opened with `#appConfirmCancelBtn` focused on desktop and 390×844 mobile.
- Pressing Enter immediately after `Delete session?` closed the dialog as a cancel action; the session remained present in DOM/API.
- Tab cycling stayed inside `#appConfirm`: desktop observed Cancel → Delete → Cancel → Delete with Tab/Tab/Shift-Tab; mobile observed Cancel → Delete → Cancel with Tab/Tab.
- Constructive `Send pending attachment?` remained confirm-focused on desktop: `#appConfirmConfirmBtn` / `Send with attachment`.
- Destructive `Reload file from disk?` opened with Cancel focused on desktop and mobile after a real file-save conflict.
- Pressing Enter immediately after `Reload file from disk?` canceled the reload: the unsaved Monaco draft remained in the editor, the conflict action remained visible, and disk content was not overwritten by the draft.
- Desktop and mobile dialog states reported no page-level horizontal overflow.
- Native `window.confirm` was overridden to count/throw; `nativeConfirmCount` stayed `0`.
- Fake broker call summary showed only state polls: `send_calls=0`, `keys_calls=0`, `shutdown_calls=0`.

## Evidence files

- Docker focused tests: `docker-focused-19490.txt` (`137 passed, 25 subtests passed`).
- Docker smoke: `docker-smoke-19491.txt` (`pre_login_api_me_status=401`, `post_login_sessions_status=200`).
- Summary assertions: `proof-summary.json` (`all_passed: true`).
- Synthetic session harness: `container/fake_confirm_focus_session.py` and `container/fake-session-start.txt`.
- API state: `api/sessions-before-browser.pretty.json`, `api/sessions-after-browser.pretty.json`.
- Desktop browser proof: `browser/desktop-delete-dialog-before-enter.json`, `browser/desktop-delete-after-enter.json`, `browser/desktop-tab-*.json`, `browser/desktop-constructive-dialog.json`, `browser/desktop-file-conflict-state.json`, `browser/desktop-reload-dialog-before-enter.json`, `browser/desktop-reload-after-enter.json`.
- Mobile browser proof: `browser/mobile-delete-dialog-before-enter.json`, `browser/mobile-delete-after-enter.json`, `browser/mobile-tab-*.json`, `browser/mobile-file-conflict-state.json`, `browser/mobile-reload-dialog-before-enter.json`, `browser/mobile-reload-after-enter.json`.
- No unintended broker mutation: `container/broker-call-summary.json`.
- Cleanup: `container/docker-stop.txt`, `container/docker-ps-after-stop.txt`.
