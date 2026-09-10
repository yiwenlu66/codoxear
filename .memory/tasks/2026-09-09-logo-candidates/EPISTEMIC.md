# Epistemic model

## Phenomenon

The previous approved paper icon deliberately included an opaque warm-gray tile so PWA and home-screen platforms can apply their own masks. Reusing that same tile for browser favicon metadata introduced visible warm-gray padding in browser chrome. The app itself did not render that approved mark: its sidebar used a separate monochrome 24px glyph, so family themes could not express brand identity.

## Mechanism

`codoxear-icon.svg/png` and `apple-touch-icon.png` remain the installation asset family: their full-bleed warm-gray raster canvases are intentional PWA padding. `favicon.svg` derives the exact approved document/fold/prompt paths, but crops the SVG viewBox to the stroked document bounds and has no tile rect; its PNG fallback therefore contains alpha outside the dog-ear instead of warm-gray pixels. `index.html` selects that SVG only for `rel="icon"`; manifest and Apple-touch metadata retain the padded installation assets.

The actual in-app brand surface is the sidebar header made by `createShellDOM` in `app_shell.js`. It now renders canonical page/fold/terminal paths with semantic child classes. `app.css` owns their base paints through `--brand-logo-*`; family stylesheet token blocks override those values and Slate adds a scoped transparent-fold structural rule. The existing theme controller remains the sole owner of `data-theme`, `data-mode`, and stylesheet swapping: logo appearance has no JavaScript writer.

## Evidence

- Docker-targeted behavioral tests (OPS correction 2026-09-10T11:06:21+08:00) passed 22 tests. They serve and parse HTML/SVG metadata, decode favicon/PWA raster pixels, execute the shell factory, and resolve theme CSS cascades.
- The favicon raster is 64×64, has alpha outside the page, reaches all bitmap bounds through its stroked geometry, and has no `#eae4d8` pixels. Apple-touch (180px) and manifest PWA (512px) corners remain `[234,228,216,255]`.
- In the Docker browser, the served page used versioned `favicon.svg` while Apple-touch and manifest stayed padded. Settings-driven live theme selection yielded Clay's terracotta/pale-fold mark, Slate's transparent-fold black outline, and Paper's approved tinted-paper palette. Screenshots are named in OPS.

## Current commitment

The branding follow-on is committed and prepared for independent review; it must not be deployed without a later explicit instruction. Its commit-archive Docker verification passed (OPS 2026-09-10T11:13:01+08:00), proving the packaged bundle boots inside an isolated Pi/broker/server/browser flow. The Docker sandbox now installs the repository's declared `tinycss2` test dependency so its standard test command can execute the existing CSS theme suite as well as this feature's behavior tests. The full isolated suite reached 1868 passed and 112 subtests; its three failures are unrelated baseline/environment boundaries documented in OPS (vendored PDF.js under the sandbox Node runtime and deploy-script clean-worktree enforcement). The prior deployment snapshot remains unchanged at `0f0e261fc163515aee705ea9ae6e64198bd25de8` until an approved follow-on is deployed.
