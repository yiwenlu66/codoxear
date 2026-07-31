# Mobile shell touch-target verification

Docker sandbox: `codoxear-mobile-shell-19385` on `http://127.0.0.1:19385/` with container HOME `/home/tester`. Fake sidecar advertised `agent_backend: cc` and pointed to a synthetic busy Claude Code JSONL log so the topbar interrupt and selected-session shell rails were visible.

## Claims exercised

1. **Phone shell controls meet the 44x44 target floor.**
   - 390x844 viewport measured visible controls: `{'#toggleSidebarBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Toggle sidebar'}, '#interruptBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Interrupt (Esc)'}, '#fileBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'View file'}, '#copyConversationBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Copy conversation'}, '#diagBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Details'}, '#unattendedBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Unattended mode'}, '#chatSearchBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Search loaded messages'}, '#prevUserBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Previous user message'}, '#nextUserBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Next user message'}, '#newBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'New session'}, '#announceBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Announcements off'}, '#notificationBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Notifications off'}}`.
   - 320x844 viewport measured visible controls: `{'#toggleSidebarBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Toggle sidebar'}, '#interruptBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Interrupt (Esc)'}, '#fileBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'View file'}, '#copyConversationBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Copy conversation'}, '#diagBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Details'}, '#unattendedBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Unattended mode'}, '#chatSearchBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Search loaded messages'}, '#prevUserBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Previous user message'}, '#nextUserBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Next user message'}, '#newBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'New session'}, '#announceBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Announcements off'}, '#notificationBtn': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Notifications off'}}`.
   - Both runs had `tooSmall=[]`.

2. **New Session backend tabs meet the same floor.**
   - 390x844 backend tabs: `{'Codex': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Codex'}, 'Pi': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Pi'}, 'Claude': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Claude'}}`.
   - 320x844 backend tabs: `{'Codex': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Codex'}, 'Pi': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Pi'}, 'Claude': {'w': 44, 'h': 44, 'visible': True, 'aria': 'Claude'}}`.
   - Codex, Pi, and Claude tabs stayed visible and selectable-sized at 44x44.

3. **The mobile shell does not create body horizontal overflow.**
   - 390x844 body geometry: `{'clientWidth': 390, 'innerWidth': 390, 'scrollWidth': 390}`.
   - 320x844 body geometry: `{'clientWidth': 320, 'innerWidth': 320, 'scrollWidth': 320}`.

4. **No backend write boundary changed.**
   - Fake broker command summary was `{'state': 47}` with `send_count=0` and `key_count=0`.
   - The proof only selected/read UI state and opened New Session; it did not send prompts or inject keys.

## Validation

- Focused local source/static suite: `42 passed, 6 subtests passed`.
- Full local suite: `1801 passed, 134 subtests passed`.
- `git diff --check` clean.
- Docker gate on port `19386`: `1800 passed, 1 skipped, 134 subtests passed`.
- Docker smoke on port `19386`: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`.

## Raw artifacts retained

- `fake_mobile_shell_session.py`
- `mobile-shell-touch-driver.js`
- `browser-mobile-shell-touch.json`
- `browser-mobile-shell-touch-320.json`
- `api-sessions-initial.json`
- `api-sessions-after-browser.json`
- `docker-calls-compact.json`
- `docker-test-19386.txt`
- `docker-smoke-19386.txt`
- `docker-final-state.txt`
