# Branding follow-on: full-bleed favicon and theme-aware in-app mark

## Objective

Keep the approved tinted-paper dog-ear as the PWA/home-screen installation icon, while making browser chrome use an unpadded, full-bleed favicon derived from the same canonical paths. Replace the legacy monochrome sidebar glyph with the approved dog-ear geometry and give the visible in-app mark a family-specific presentation through CSS/theme tokens.

## Contract

- `codoxear/static/codoxear-icon.svg` and its 512px PNG remain the padded, opaque warm-gray PWA/manifest asset. `apple-touch-icon.png` remains that padded installation treatment.
- `favicon.svg` and `favicon.png` are the distinct browser assets. They use the approved document/fold/terminal paths and colors without a warm-gray outer canvas. The favicon SVG is the HTML `rel="icon"` target; PNG remains the compatibility route and `/favicon.ico` fallback.
- The app sidebar header is the product’s in-app brand surface. It renders semantic page/fold/terminal paths with `--brand-logo-*` tokens; theme selection remains solely owned by the existing theme controller and stylesheet cascade.
- Paper light preserves the approved tinted-paper rendering. Clay and Slate visibly differ through their family tokens; Slate also presents the fold as a monochrome outline. No JavaScript writes logo appearance.
- Browser/runtime checks run only against Docker isolation. Deployment remains pending parent review for this follow-on.

## Success criteria

1. Served HTML keeps favicon and PWA/home-screen metadata distinct and versioned.
2. The favicon raster has no warm-gray outer-canvas pixels, while Apple-touch and manifest raster corners remain opaque warm gray.
3. The sidebar renders the approved dog-ear mark, and theme selections yield three distinct computed logo presentations.
4. Behavioral tests execute HTML/SVG delivery, image decoding, shell DOM construction, and CSS token cascade; Docker browser evidence covers live theme switches.
5. The change is committed atomically after reviewable staged-diff inspection; deployment is not run.
