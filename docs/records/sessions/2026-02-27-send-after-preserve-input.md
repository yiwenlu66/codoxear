# Session: 2026-02-27 Send-After Preserve Input

## Focus
Fix `Send after current` losing composer text when queue push does not complete.

## Requests
- After clicking `Send after current`, text should not disappear unless queueing actually succeeds.
- Prevent queue action from drifting to another session if selection changes during async work.

## Actions Taken
- Updated `queueServerMessage` to:
  - accept an explicit session id override,
  - return a success boolean.
- Updated send-choice `Send after current` click path to:
  - capture `sid` at click time,
  - queue against that `sid`,
  - clear composer only when queue push returns success.
- Updated composer clear helper to support session-aware clearing (`clearComposer(sid)`), so async queue completion does not wipe the wrong session's draft.

## Outcomes
- Failed queue push no longer drops the typed message.
- Successful queue push still clears composer as before.
- Queueing from send-choice now targets the intended session deterministically.

## Tests
- `node --check codoxear/static/app.js`
- `python3 -m unittest discover -s tests`
