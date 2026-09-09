# Epistemic model

## Phenomenon
The user wants the original Codoxear folded-paper terminal mark, rendered in B Tinted Paper's warm palette. The prior combination was wrong because it retained B's slender document and compact rounded prompt while increasing stroke width; it therefore did not match the reference's visibly broader, bolder silhouette.

## Mechanism
The reference's perceived fatness is geometry, not stroke weight alone: its page is the exact `M142 104h172l88 88v216H142z` document path and its prompt is the exact square-capped, miter-joined `m204 268 38 36-38 36M270 340h50` path, both at 24px. Copying those paths and their stroke attributes makes the geometry mechanically identical. Recoloring their existing fill/stroke regions leaves geometry intact. Removing the reference's second, inset tile rectangle meets the one-tile requirement.

## Current evidence
`review/logo-paper-refined/reference-geometry-tinted-paper.svg` has exactly one rounded tile (`#eae4d8`) and copies every document/prompt `d`, 24px stroke width, line cap, and line join from `review/logo-candidates-2026-09-09/01-paper-outline.svg`; it recolors only the page (`#fdfbf6`), contour/fold (`#a79a84`), existing fold fill (`#e8e1d2`), and prompt (`#c96442`). XML comparison asserts those geometry attributes equal the source and confirms one tile rect. Its 512px PNG and comparison sheet were visually inspected at 180px and 60px; the two marks have the same broad page and large bold `>_`, with only palette and the removed inner reference border differing. See OPS 2026-09-10T00:44:41+08:00.

## Current commitment
This is one corrected preview, not a new candidate set. The authoritative deliverable is `review/logo-paper-refined/reference-geometry-tinted-paper.png`; `reference-geometry-tinted-paper-comparison.png` places source and recolor side by side at 180px and 60px. Production branding remains untouched.

## Dead directions
Do not return to B's narrower rounded page, its smaller round-capped prompt, or any invented redraw. Abstract non-page metaphors are rejected by the user.

## Remaining uncertainty
None for the requested geometry correction. No trademark clearance was performed.
