# Epistemic model

## Phenomenon
The approved Codoxear mark is the original broad folded-paper terminal geometry recolored in the tinted-paper palette. Production must reproduce that precise visible mark without depending on exploratory review files and must provide correct installation assets.

## Mechanism
`codoxear/static/codoxear-icon.svg` owns the vector paths and palette; all installed PNGs are rasterizations at their platform sizes. Its background is intentionally a full-bleed opaque `#eae4d8` rectangle, so platform masks round the icon instead of exposing transparent/black corners. The static asset-version registry includes the source and each raster output, and the served HTML projects versioned favicon, apple-touch, and manifest links. The manifest resolves its relative `codoxear-icon.png` URL from `/manifest.webmanifest` to the top-level static route.

## Evidence
The production SVG's three page/prompt paths, colors, 24px strokes, caps, and joins compare equal to the approved review SVG; only outer-tile rounding and accessible metadata differ. Docker tests and browser/API evidence in OPS 2026-09-10T00:54:13+08:00 confirm all links, dimensions, opacity, manifest resolution, and rendered 512px screenshot (`/tmp/codoxear-logo-docker-icon.png`). The committed Git archive also passed `scripts/docker_verify.sh` in an isolated Pi/broker/browser flow (OPS 2026-09-10T00:56:23+08:00).

## Constraint
The visible non-beige foreground radius is 222.85px of the 512px icon, outside the common 204.8px mask-safe radius. The manifest deliberately omits a maskable purpose. Dynamic app theme-color remains owned by the existing theme code; the manifest's fixed PWA colors describe the approved app icon background only.

## Current commitment
The promotion awaits independent review and explicit authorization. Do not deploy in this phase. iOS may retain an existing apple-touch icon until its cache is invalidated by the versioned HTML link and a home-screen shortcut refresh/re-add.
