# Web PWA Support Design

**Date:** 2026-04-09

## Goal

Add a lightweight, deployment-safe PWA layer to `web/` so the app installs cleanly on mobile and desktop, exposes correct metadata to browsers, and keeps the existing push-notification service worker behavior intact.

This change is intentionally scoped to the user's approved "basic" PWA tier.

## Approved Direction

Keep the existing custom service worker and improve the installability layer around it.

User-approved decisions:

- Keep `web/public/service-worker.js` as the single service worker
- Do not add offline caching or app-shell guarantees in this pass
- Do not replace the current push notification logic
- Improve manifest quality, registration correctness, and mobile install compatibility
- Limit code changes to PWA-related files in `web/`

## Problem Statement

`web/` already contains partial PWA pieces, but the implementation is incomplete and fragile.

Observed issues:

- `web/public/manifest.webmanifest` is minimal and does not include the richer metadata browsers expect for polished installation
- `web/index.html` hard-codes root-relative PWA asset links, which can break when the app is served under `CODEX_WEB_URL_PREFIX`
- `web/src/main.tsx` registers `service-worker.js` with a relative string instead of using the Vite base-aware asset path
- iOS-specific install metadata is incomplete
- The current setup suggests PWA support, but the install path is not fully normalized across browsers and deployment prefixes

## In Scope

- Make manifest and service worker URLs work correctly with Vite's `base` setting
- Improve install metadata in `web/index.html`
- Expand manifest fields to better support Android and desktop installation
- Ensure service worker registration remains compatible with the current push notification features
- Add or generate the icon assets needed for install surfaces where practical
- Add targeted test coverage for the updated registration behavior if current tests cover that area

## Out of Scope

- Offline page or offline app-shell support
- Precaching static assets
- Runtime caching strategies for API or media content
- Update prompts, background sync, or advanced PWA lifecycle UI
- Replacing the current custom service worker with `vite-plugin-pwa`

## Why This Direction

This is the smallest change set that materially improves PWA support without destabilizing the existing app.

- It preserves the current notification-related worker behavior
- It avoids a service worker migration while the repo already has substantial unrelated local changes
- It fixes correctness issues that would affect installation under prefixed deployments
- It leaves a clean upgrade path to `injectManifest` later if offline support becomes a requirement

## Design Principles

1. **One service worker, one responsibility boundary**
   - Keep the current custom service worker as the runtime worker.
   - Do not introduce a second worker generator in this change.

2. **Base-path correctness first**
   - Any manifest or worker URL emitted by the app must respect the Vite `base` path.
   - Prefixed deployments must behave the same as root deployments.

3. **Installability over offline ambition**
   - Browsers should recognize the app as installable with complete metadata.
   - The app should not claim offline behavior it does not implement.

4. **Preserve current push semantics**
   - Push event handling and notification-click behavior remain intact.
   - PWA polish must not break notification enrollment flows already covered by the app.

## Architecture

### 1. Manifest stays static but becomes richer

`web/public/manifest.webmanifest` remains the manifest source, but its content is upgraded.

Expected improvements:

- add a clearer `description`
- ensure `name` and `short_name` stay stable
- make `start_url` and `scope` consistent with the current deployment model
- keep `display: "standalone"`
- add `orientation` only if it matches current app usage; otherwise omit it
- add multiple icon entries where available, including at least a maskable icon entry if we can produce one cleanly from current assets

Because files under `public/` are copied as-is by Vite, manifest-relative asset paths remain stable as long as the HTML references the manifest through a base-aware URL.

### 2. HTML metadata becomes base-aware and mobile-friendly

`web/index.html` should stop hard-coding root-relative PWA asset links.

Changes:

- use Vite base-aware references for the favicon and manifest
- add Apple mobile web app metadata
- add `apple-touch-icon`
- keep `theme-color` aligned with the manifest theme color

This ensures PWA metadata works when the app is served from `/` or from a prefixed path such as `/foo/`.

### 3. Service worker registration becomes explicit and deployment-safe

`web/src/main.tsx` currently registers `service-worker.js` using a bare string.

New rule:

- resolve the worker URL through `import.meta.env.BASE_URL`
- register on window load as today
- keep failure handling silent unless there is an existing app-level reporting pattern worth reusing

This preserves current behavior while preventing path bugs in prefixed deployments.

### 4. Existing service worker logic remains unchanged unless path handling needs a small correction

`web/public/service-worker.js` already handles:

- `push`
- `notificationclick`

The worker should remain functionally focused on notifications.

This pass should only touch it if one of these is needed:

- a path fix tied to base/scope behavior
- a tiny metadata-safe cleanup required by browser install surfaces

Otherwise, leave its event logic unchanged.

## Asset Strategy

The app already ships:

- `web/public/favicon.png`
- `web/public/codoxear-icon.png`

Planned asset behavior:

- keep existing icon assets if they satisfy the install surfaces
- add one or more derived icon files only if needed for manifest completeness or Apple install polish
- prefer simple PNG outputs over adding a new icon-generation toolchain in this pass

If the existing 512x512 icon is the only clean source asset, it is acceptable to reuse it for both general and maskable entries in the basic tier, as long as the manifest stays valid.

## Testing Strategy

Because this is a behavior/configuration change, testing should stay targeted.

Planned verification:

- run the `web/` test suite if existing tests cover service worker registration paths
- add or update a focused unit test around the service worker registration URL if needed
- run a production build to confirm the app emits the manifest and service worker assets without path regressions

Manual spot-check goals:

- built HTML references the manifest correctly
- built app still registers the service worker
- notification-related code paths are not broken by the registration change

## File-Level Plan

Primary files likely to change:

- `web/index.html`
- `web/src/main.tsx`
- `web/public/manifest.webmanifest`
- `web/public/service-worker.js` only if a small path-safe adjustment is required
- one or more files under `web/public/` for icon assets if needed
- relevant tests in `web/src/app/` or nearby only if registration coverage must be updated

## Risks and Mitigations

### Risk: prefixed deployments still resolve an asset incorrectly

Mitigation:

- use Vite base-aware URLs from HTML and runtime code
- confirm with a production build artifact review

### Risk: install metadata improves, but iOS still has inconsistent icon behavior

Mitigation:

- add the explicit `apple-touch-icon` tag
- keep icon filenames simple and PNG-based

### Risk: changing the worker registration path breaks push subscription flows

Mitigation:

- keep the same registration timing and same worker file
- only change URL construction, then run targeted tests and build verification

## Future Upgrade Path

If stronger PWA support is needed later, the next step should be a controlled migration to `vite-plugin-pwa` with `injectManifest`, not `generateSW`.

That future pass would allow:

- precache support
- offline shell behavior
- cleaner update handling
- preservation of the custom push event logic inside a build-managed worker

## Acceptance Criteria

This basic PWA pass is complete when:

- the app remains installable in browsers that already support install prompts
- manifest and favicon links resolve correctly under both root and prefixed deployments
- service worker registration uses a base-aware URL
- current push notification registration behavior still works
- no offline-support claims are introduced into the UI or config
