# Visual findings — Codoxear logo exploration

## Findings

1. **High — the production mark is compositionally overfilled.**
   `codoxear/static/codoxear-icon.png` places a heavy dog-ear outline almost edge to edge and lets the `>` compete with the page silhouette. In the supplied comparison screenshot, this produces the right-hand mark’s cramped, oversized, rough-looking glyph. The 64px `favicon.png` repeats the same proportions.

2. **Medium — the older mark has the right semantic hierarchy but the wrong rendering language.**
   The historical icon from `0487b711` / `ea995651` is immediately recognizable as “document containing a terminal.” Its blue-purple gloss, translucent streaks, lifted paper, and drop shadow are at odds with Codoxear’s current flat paper, clay, and slate families. The later flat versions (`8f2cfb39`, `9fe17f82`) correctly removed those effects but made the page outline and prompt too large.

3. **Info — the proposed set restores the hierarchy through scale, not detail.**
   Each candidate reserves a rounded-tile safe area, makes the page the primary silhouette, and confines terminal geometry to a compact two-stroke `>_` cue. There are no gradients, filters, shadows, or sub-24-unit terminal strokes. The contact sheet’s rasterized 64px and 180px instances are visually legible.

4. **Info — production references remain untouched.**
   `codoxear/static/codoxear-icon.png`, `codoxear/static/favicon.png`, `codoxear/static/index.html`, and `codoxear/static/manifest.webmanifest` were inspected but not modified. All exploratory work is isolated in this directory.

## Recommended review order

- **Start with 02 — Clay Solid** if the app icon should advertise the default warm editorial identity.
- **Compare 03 — Slate Monogram** where high contrast and a neutral platform-adjacent look matter most.
- **Keep 01 — Paper Outline** as the closest, conservative successor to the original document-plus-terminal identity.
- **Use 04 or 05** only if a substantially darker or bolder home-screen presence is desired.

## Residual risks

- iOS applies its own mask and may alter the perceived corner radius and edge margin. A selected candidate needs an installed-device home-screen check before it replaces production icon and favicon files.
- The best candidate depends on a home-screen context (wallpaper, neighboring icons, light/dark icon treatment) that the supplied screenshot does not fully establish.
- The present review validates geometry and raster legibility at 64px/180px, not all platform export requirements. The chosen SVG will still need deliberate PNG/favicons/manifest variants at the production handoff.
