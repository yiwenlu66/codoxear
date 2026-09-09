## Objective
Promote the user-approved `review/logo-paper-refined/reference-geometry-tinted-paper.svg/png` mark to Codoxear production branding, then leave the resulting commit for independent review before any deployment.

## Approved design contract
- Exact original Paper Outline page and terminal geometry: document `M142 104h172l88 88v216H142z`; terminal `m204 268 38 36-38 36M270 340h50`; 24px strokes with the approved joins/caps.
- Approved tinted-paper palette: beige `#eae4d8` canvas, cream `#fdfbf6` page, taupe `#a79a84` contour/fold, `#e8e1d2` fold fill, terracotta `#c96442` terminal.
- The canonical production source must be a static SVG owned by the application, never the review directory.
- Installed raster outputs: 512px `codoxear-icon.png`, existing-target-size `favicon.png` (64px), and 180px `apple-touch-icon.png`. Each must be opaque at every corner with full-bleed beige canvas; platforms, not transparent pixels, provide rounded masks.
- Add the apple-touch link, use resolved production manifest icon URLs, update stale manifest blue branding colors, and retain dynamic UI `theme-color` ownership.
- Do not declare a maskable icon: the approved foreground exceeds the standard 80%-diameter safe circle.

## Verification and release boundary
- Verify only in Docker isolation. Required evidence includes served landing HTML links with versioned URLs, asset fetching and dimensions from a browser, manifest icon resolution, favicon link projection, and a browser screenshot of the served icon.
- Do not deploy in this phase. The next parent must independently review the committed diff and explicitly authorize deployment.
