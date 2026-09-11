# Paper-outline refinement round

The user selected round-one **01 Paper Outline** and rejected all abstract directions. This round refines only the drawing quality of that mark: lighter contour, softly rounded page corners, a smaller prompt near mid-left, one tile surface (the reference's inner stroked tile border is gone), and calmer page proportions (~0.73 width/height, ~60% of tile height).

- `reference-01-paper-outline.svg` — unchanged copy of the selected round-one candidate.
- `A-slim-outline.svg` — slim charcoal outline, rounded page corners, small rust prompt on cream.
- `B-tinted-paper.svg` — lightly tinted filled page, delicate taupe contour, quiet tonal fold, warm neutral tile.
- `C-monoline.svg` — unfilled continuous monoline contour; document, fold, and prompt share one equalized stroke weight.
- `D-terracotta-contour.svg` — warm terracotta contour and pale clay fold with a small charcoal prompt.
- `E-old-flattened.svg` — flat reconstruction of the original icon: unstroked white page, tonal fold, very small prompt, warm gray tile.

`comparison.png` shows all six through the same baked iOS-like rounded mask at 170px and 60px. Run `python3 build.py` to regenerate every PNG (uses `rsvg-convert`). Production assets are unchanged.
