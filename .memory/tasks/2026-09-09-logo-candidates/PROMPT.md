## Objective
Refine the user-selected **01 Paper Outline** direction into polished candidates. The user rejected the fresh abstract round ("all very ugly") and explicitly prefers the paper-outline mark. The work is complete when a labeled visual comparison of refined paper variants is available and each can be judged at realistic iOS icon sizes.

## Workbench
- Keep the folded document + small terminal prompt identity; refine drawing quality only.
- Preserve cream/warm-gray/terracotta palette, generous margins, one tile surface (no inner tile border).
- Page height ~58-64% of tile, prompt ~25-35% of page width near midheight, contour lighter than round-one 01 but legible at 60px.
- No gradients, bevels, shadows, large prompts, black-heavy tiles, or abstract symbols.
- Present candidates for user selection before changing production assets.

## Current round
`review/logo-paper-refined/`: unchanged reference + A (slim outline), B (tinted paper), C (monoline), D (terracotta contour), E (flattened original); `build.py` renders 180/60px PNGs and the 2×3 `comparison.png`. Prior rounds preserved in `review/logo-candidates-2026-09-09/` and `review/logo-candidates-fresh-2026-09-09/`.

## Current follow-up
The user requested one combination preview only: retain **B · Tinted Paper** geometry and palette (beige tile, cream page, taupe contour/fold, terracotta small mid-left terminal) while adopting the selected original reference's heavier line treatment. Deliver a new named standalone combined SVG and 512px rounded-tile PNG, plus a labeled B-original versus heavier-combined preview at 180px and 60px. Preserve all earlier candidates and do not modify production assets.

## Context
- Repository: `/home/yiwen/codoxear`
- User comparison image: `/home/yiwen/.local/share/codoxear/uploads/broker-118886/1788937065328_IMG_3096.jpeg`
- Old logo is the left icon; current logo is the right icon.
- Current production icon: `codoxear/static/codoxear-icon.png`.

## Task specifications
- The old logo's document-with-terminal identity is the baseline.
- Reduce the old logo's glossy/translucent rendering and align color treatment with the current clay/slate/paper themes.
- Correct the current logo's shabby execution and oversized glyph.
- Produce several distinct candidates, not one recommendation presented multiple ways.
- Show candidates in one labeled comparison at large and small scale.
- Do not replace the production icon until the user chooses a direction.

## Constraints
- Keep candidates legible at iOS home-screen size.
- Avoid fragile micro-detail and excessive gradients.
- Preserve a clear terminal/document cue.
- Use deterministic vector construction where practical.
- Do not overwrite production assets during exploration.
- This follow-up is a single requested combination, not a new option set: preserve B proportions, rounded page corners, prompt placement, and one-tile construction. Raise contour, fold, and prompt strokes to the original reference's 24px treatment; alter only the fold/prompt gaps needed for that weight to remain legible.
