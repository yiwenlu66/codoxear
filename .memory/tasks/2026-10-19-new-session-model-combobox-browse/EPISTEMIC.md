# EPISTEMIC — new-session provider/model combobox browse

## Phenomenon

The New session dialog's combined "Provider / model" field is prefilled with a
committed pair rendered `<model> · <provider>`. Opening its list or typing into
it made the field's own text act as the filter query, so the picker opened on
exactly the current model (plus ids it prefixes) and typing appended to the
display string, producing an unparseable pair. The user had to hand-clear the
field before any other model was reachable.

## Mechanism (settled)

One field carries two different kinds of content — a committed selection and a
user query — and the list filter read both as a query. Everything else follows
from separating them:

- browse ordering is a *directory* concern (recents, then provider groups),
  while filtering is a *ranking* concern (exact/prefix/contains, cap 12);
- a committed pair is atomic and replaced wholesale on the first edit, because
  a partial edit of `model · provider` is never a meaningful query;
- a browse-only empty is a UI state, so it must be restored, and the restore
  must also apply inside `start()` because launch parses the field.

Confirmed by 19 browser checks against a committed Docker build, including the
end-to-end launch record: browse-clear then Start launched the committed pair,
not `defaults.model` (the silent-failure mode this design exists to prevent).

## Live mechanisms / ruled out

- Ruled out: keeping the text and filtering on a "browse" flag. It cannot
  satisfy "clear the previous string", and the field would then display a
  selection while the list shows a different set.
- Ruled out: deriving the restore value from `(provider(), defaults.model)`.
  A launch preset can commit a pair that does not match the backend default, so
  that recomputation silently changes the launch target. The restore value must
  be the exact text captured at clear time.
- Ruled out: `onfocus` select-all as the replace mechanism. A mouse click places
  the caret after the focus event, so it only works for keyboard focus; a
  `beforeinput` clear covers keyboard, pointer focus, and paste with one rule.

## Residual risks

- `beforeinput` on iOS with an IME composition that starts on a committed pair:
  the handler clears on `insertCompositionText` like any insert. Tap-to-clear
  already covers mobile entry, and the desktop Chrome path is verified; an
  iOS-specific composition quirk is unverified and untestable in this harness.
- List order for a provider whose models are unordered config: order follows
  config order (deliberate — Pi's models.json order is meaningful).
- Unchanged, pre-existing: typing a bare model id (no provider) resets the
  provider choice to `defaultNewSessionProviderChoice()`. Now reachable by
  browsing and typing rather than only by manual clearing; not addressed here
  because it is a parse semantic, not a browse one.
