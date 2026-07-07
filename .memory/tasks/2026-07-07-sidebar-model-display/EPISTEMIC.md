# EPISTEMIC

## Phenomenon
The sidebar is the primary session navigation surface. Backend and reasoning effort are visible there, but model identity is absent even though it is available in the session-list payload and matters for distinguishing same-project sessions.

## Current mechanism
`codoxear/static/app.js` builds `.sessionMetaLine` with backend logo, owner icon, optional effort marker, and `.metaText`. The `.metaText` string currently concatenates age, cwd base name, and git branch; it does not read `s.model`. Server/session listing already includes `model`, and Details/diagnostics can show it.

## Current claim
This is a display-projection gap, not a backend or API gap. The fix should project meaningful `s.model` values into the existing `.metaText` sequence and rely on existing ellipsis behavior for tight/mobile layouts.

## Key uncertainty
The only design boundary is noise suppression: omit absent/empty/`default` model values, but show explicit non-default model strings, including provider/model literals.
