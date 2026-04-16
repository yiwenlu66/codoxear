# Web PWA Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `web/` install cleanly as a basic PWA by fixing manifest/meta completeness and making service-worker-related asset URLs respect the Vite base path.

**Architecture:** Add a tiny shared helper for base-aware public asset URLs, reuse it from both app boot and push-notification registration code, and keep the existing custom service worker in place. Update the static manifest and HTML metadata for install surfaces without introducing offline caching or a new worker toolchain.

**Tech Stack:** Vite 7, Preact, TypeScript, Vitest, static assets in `web/public/`

---

## File Structure

- `web/src/lib/publicAssetUrl.ts` — single responsibility: join `import.meta.env.BASE_URL` with a public asset path safely
- `web/src/lib/publicAssetUrl.test.ts` — unit coverage for root and prefixed deployments
- `web/src/main.tsx` — app boot path; register the existing service worker with a base-aware URL
- `web/src/app/AppShell.tsx` — push-notification path; reuse the same base-aware service worker URL
- `web/src/app/AppShell.test.tsx` — verify push registration still occurs and uses the expected worker URL
- `web/index.html` — PWA metadata tags and base-aware manifest/icon links
- `web/public/manifest.webmanifest` — richer manifest fields for installability
- `web/public/codoxear-icon.png` / `web/public/favicon.png` — existing source assets reused for icon references unless a derived icon file becomes necessary

## Task 1: Add failing tests for base-aware public asset URLs

**Files:**
- Create: `web/src/lib/publicAssetUrl.test.ts`
- Modify: `web/src/app/AppShell.test.tsx`
- Test: `web/src/lib/publicAssetUrl.test.ts`
- Test: `web/src/app/AppShell.test.tsx`

- [ ] **Step 1: Write the failing unit test for the asset URL helper**

```ts
import { describe, expect, it } from "vitest";
import { toPublicAssetUrl } from "./publicAssetUrl";

describe("toPublicAssetUrl", () => {
  it("keeps root deployments rooted at slash", () => {
    expect(toPublicAssetUrl("service-worker.js", "/")).toBe("/service-worker.js");
    expect(toPublicAssetUrl("/manifest.webmanifest", "/")).toBe("/manifest.webmanifest");
  });

  it("prefixes assets for nested deployments", () => {
    expect(toPublicAssetUrl("service-worker.js", "/codoxear/"))
      .toBe("/codoxear/service-worker.js");
    expect(toPublicAssetUrl("/favicon.png", "/nested/app/"))
      .toBe("/nested/app/favicon.png");
  });
});
```

- [ ] **Step 2: Run the helper test to prove it fails before implementation**

Run: `cd web && npm test -- --run web/src/lib/publicAssetUrl.test.ts`
Expected: FAIL with `Cannot find module './publicAssetUrl'` or equivalent missing-export error.

- [ ] **Step 3: Tighten the existing AppShell push test so it checks the root-base worker URL**

Keep the AppShell red-phase coverage focused on the current integration seam by updating the existing mobile push notification test in `web/src/app/AppShell.test.tsx` to assert the exact register call for the root-base case:

```ts
expect(register).toHaveBeenCalledWith("/service-worker.js");
```

The prefixed-base behavior is already covered in `web/src/lib/publicAssetUrl.test.ts`, which keeps Task 1 realistic before the production helper exists.

- [ ] **Step 4: Run the focused AppShell test to confirm the new assertion fails**

Run: `cd web && npm test -- --run web/src/app/AppShell.test.tsx`
Expected: FAIL because the code still calls `navigator.serviceWorker.register("service-worker.js")`.

- [ ] **Step 5: Review the diff before implementation**

Run: `git diff -- web/src/lib/publicAssetUrl.test.ts web/src/app/AppShell.test.tsx`
Expected: only new failing test coverage for URL construction and service-worker registration.

## Task 2: Implement the shared asset URL helper and wire service worker registration through it

**Files:**
- Create: `web/src/lib/publicAssetUrl.ts`
- Modify: `web/src/main.tsx`
- Modify: `web/src/app/AppShell.tsx`
- Test: `web/src/lib/publicAssetUrl.test.ts`
- Test: `web/src/app/AppShell.test.tsx`

- [ ] **Step 1: Implement the helper with explicit base normalization**

Create `web/src/lib/publicAssetUrl.ts` with:

```ts
const trimLeadingSlashes = (value: string) => value.replace(/^\/+/, "");
const trimTrailingSlashes = (value: string) => value.replace(/\/+$/, "");

export function toPublicAssetUrl(path: string, baseUrl = import.meta.env.BASE_URL): string {
  const assetPath = trimLeadingSlashes(path);
  const normalizedBase = trimTrailingSlashes(String(baseUrl || "/").trim()) || "";
  return normalizedBase ? `${normalizedBase}/${assetPath}` : `/${assetPath}`;
}
```

- [ ] **Step 2: Update app boot to use the shared helper**

Change `web/src/main.tsx` to:

```ts
import { toPublicAssetUrl } from "./lib/publicAssetUrl";

installViewportCssVars();
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register(toPublicAssetUrl("service-worker.js")).catch(() => undefined);
  });
}
```

- [ ] **Step 3: Update push-notification registration to use the same helper**

Change the worker registration in `web/src/app/AppShell.tsx` to:

```ts
import { toPublicAssetUrl } from "../lib/publicAssetUrl";

const ensureVoiceServiceWorker = async () => {
  if (!("serviceWorker" in navigator)) {
    throw new Error("service workers are not available");
  }
  return navigator.serviceWorker.register(toPublicAssetUrl("service-worker.js"));
};
```

- [ ] **Step 4: Run the focused tests and make sure they pass**

Run: `cd web && npm test -- --run web/src/lib/publicAssetUrl.test.ts web/src/app/AppShell.test.tsx`
Expected: PASS for the helper coverage and the updated push-registration assertions.

- [ ] **Step 5: Review the runtime diff for only the intended registration changes**

Run: `git diff -- web/src/lib/publicAssetUrl.ts web/src/main.tsx web/src/app/AppShell.tsx`
Expected: a small helper file plus two call sites using it.

## Task 3: Upgrade HTML metadata and manifest for basic PWA installability

**Files:**
- Modify: `web/index.html`
- Modify: `web/public/manifest.webmanifest`
- Test: `web/package.json` build script via `npm run build`

- [ ] **Step 1: Add the missing metadata tags in `web/index.html`**

Update the head block to use Vite base-aware public URLs and iOS metadata:

```html
<link rel="icon" type="image/png" href="%BASE_URL%favicon.png" />
<link rel="apple-touch-icon" href="%BASE_URL%codoxear-icon.png" />
<link rel="manifest" href="%BASE_URL%manifest.webmanifest" />
<meta name="theme-color" content="#1d4ed8" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="default" />
<meta name="apple-mobile-web-app-title" content="Codoxear" />
```

Keep the existing viewport settings and HLS script unchanged.

- [ ] **Step 2: Expand the manifest to cover basic install surfaces**

Replace the current manifest contents with:

```json
{
  "name": "Codoxear",
  "short_name": "Codoxear",
  "description": "Continue and monitor local CLI agent sessions from a browser.",
  "start_url": "./",
  "scope": "./",
  "display": "standalone",
  "background_color": "#e9eef5",
  "theme_color": "#1d4ed8",
  "icons": [
    {
      "src": "favicon.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "codoxear-icon.png",
      "sizes": "512x512",
      "type": "image/png"
    },
    {
      "src": "codoxear-icon.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "maskable any"
    }
  ]
}
```

If browser install testing shows the 192x192 entry is too small or visually poor, derive a dedicated `192x192` PNG from the existing source asset in `web/public/` and update the `src` accordingly.

- [ ] **Step 3: Run the web test suite as a regression check**

Run: `cd web && npm test`
Expected: PASS, confirming the registration change did not break existing UI behavior.

- [ ] **Step 4: Run a production build and inspect the emitted assets**

Run: `cd web && npm run build`
Expected: PASS, followed by built output under `web/dist/` and copied assets under `codoxear/static/dist/`.

Then inspect the key outputs:

```bash
cd web && rg -n "manifest|apple-touch-icon|theme-color" dist/index.html
cd web && ls dist | rg "manifest|service-worker|favicon|icon"
```

Expected:
- `dist/index.html` references the manifest and Apple icon with the correct base prefix
- `dist/manifest.webmanifest` exists
- `dist/service-worker.js` exists

- [ ] **Step 5: Review the final PWA-only diff**

Run: `git diff -- web/index.html web/public/manifest.webmanifest web/src/lib/publicAssetUrl.ts web/src/lib/publicAssetUrl.test.ts web/src/main.tsx web/src/app/AppShell.tsx web/src/app/AppShell.test.tsx`
Expected: only the lightweight PWA changes from this plan.

## Self-Review Checklist

- Spec coverage: the plan covers base-aware URLs, richer install metadata, preserved push worker behavior, and verification through tests plus a production build
- Placeholder scan: no `TODO`, `TBD`, or "write tests" placeholders remain
- Type consistency: the shared helper name is `toPublicAssetUrl` everywhere in the plan, and both runtime call sites use the same `service-worker.js` asset path

## Notes For Execution

- Do not introduce offline caching or a second service worker in this plan
- Do not use `vite-plugin-pwa` in this pass
- Do not commit automatically during execution because the worktree already contains unrelated local changes; keep review focused on file-scoped diffs instead
