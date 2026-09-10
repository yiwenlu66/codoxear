# Epistemic model

## Phenomenon

The installation icon needs an opaque warm-gray canvas so home-screen platforms can mask it; the browser favicon must instead show the approved dog-ear without that canvas. The first separation implementation correctly retained distinct assets but carried the PWA coordinate system into the 20px sidebar mark and stretched the non-square favicon raster fallback.

## Mechanism

`codoxear-icon.svg/png`, `apple-touch-icon.png`, and `manifest.webmanifest` remain the padded installation family. `favicon.svg` contains the approved document/fold/prompt paths in the tight `130 92 284 328` viewport without a background rect. Its PNG fallback is made by uniformly rasterizing the SVG at 56×64 and centering it in a transparent 64×64 canvas, so the fallback does not horizontally deform the mark.

The sidebar is the in-app brand surface. Its semantic dog-ear SVG uses that same tight viewport, which maps its 328-unit height onto the existing 20px CSS height and yields a 17.317px painted width. The CSS theme-token design remains unchanged: `app.css` owns base `--brand-logo-*` paints and family stylesheets supply Clay/Slate/Paper presentation without a JavaScript appearance writer.

## Evidence

- The favicon fallback alpha bbox is `(4, 0, 60, 64)`: vertical paint fills the 64px raster, 4px transparent side margins preserve a 56/64 aspect of 0.875, and the difference from the SVG viewport aspect 284/328 (~0.866) is below 0.02. It has no `#eae4d8` pixels.
- Docker targeted tests pass 22 tests, including the served HTML/SVG/image decode test that compares PNG painted aspect against the parsed SVG viewport and the executed shell DOM test that pins the tight sidebar viewport.
- Docker browser evidence on port 19011 measured the sidebar as a 20px square containing a 20px-tall, 17.317px-wide mark. Clay, Slate, and Paper stayed visually distinct in screenshots named in OPS.
- PWA/Apple assets and manifest are byte-identical to the reviewed parent commit.

## Current commitment

The two independent-review blockers are fixed in the working successor change. It has not been deployed; parent review determines the next release action. The sandbox wrapper's root-owned HOME failure and the three full-suite failures are environmental/unrelated boundaries described in OPS: deploy fixture needs a writable repository for Git worktree creation, and the vendored PDF.js Node checks need `Promise.withResolvers`, unavailable in the sandbox's Node 20.19.2.
