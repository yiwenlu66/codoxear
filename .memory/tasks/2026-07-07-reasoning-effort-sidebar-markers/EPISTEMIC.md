# EPISTEMIC

## Phenomenon
Sidebar session rows are the quick scan surface for backend/model/reasoning state. Some supported reasoning efforts are currently invisible there: `max`, `minimal`, and `off` are valid session values, but the marker mapping returns an empty string for them.

## Current mechanism
`codoxear/static/app.js` computes `effortTxt` from `s.reasoning_effort` and maps only four values: `xhigh -> X`, `high -> H`, `medium -> M`, `low -> L`. It only appends `.effortMark` when the mapped string is non-empty. `codoxear/static/app.css` has color rules for those four classes only.

## Current claim
This is a bounded display-truth gap. Launch/default plumbing for these efforts already exists; the defect is that the sidebar suppresses valid values. The fix should be a frontend display mapping/CSS/test/proof slice, not a launch or backend change.

## Key uncertainty
Choose compact glyphs that are legible in the existing 12px marker space without confusing `max` with `medium`; then prove actual browser sidebar rows show marker text and titles for representative supported values.
