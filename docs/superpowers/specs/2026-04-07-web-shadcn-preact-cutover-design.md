# Web Shadcn-Preact Cutover

## Goal

Rebuild the `web/` frontend around a `shadcn-preact`-style component system while keeping the existing `Preact + Vite + TypeScript` runtime, current API contracts, and current store-driven product behavior.

This is a full UI cutover for the current web client, not a light restyle.

## Current State

The current `web/` frontend already has the right broad architecture:

- `web/src/app/` owns the application shell and providers.
- `web/src/domains/*` owns the main session, message, composer, and UI state.
- `web/src/components/*` owns the rendered surfaces.
- `web/src/lib/*` owns API and shared helpers.

The main weakness is the UI layer, not the product model.

Today the frontend is still dominated by large global styles in:

- `web/src/styles/global.css`
- `web/src/styles/theme.css`

The result is that the app is component-based in code, but not yet component-system-based in presentation. Layout, spacing, color, and surface patterns are still defined mostly through page-specific CSS rather than reusable UI primitives.

## Chosen Direction

Adopt `shadcn-preact` as the UI foundation for `web/` and perform a one-shot visual and structural cutover of the main rendered surfaces.

This means:

- keep `Preact`
- keep `Vite`
- keep the current domain stores and API shape
- add `Tailwind CSS 3` and the token model expected by `shadcn-preact`
- copy in project-owned UI primitives under `web/src/components/ui/`
- rewrite the current page surfaces to compose those primitives instead of relying on large bespoke CSS blocks

This does **not** mean migrating to React or directly depending on the official React `shadcn/ui` package flow.

## Why This Direction

`shadcn-preact` matches the project more directly than a React migration:

- it is designed for `Preact + Vite`
- it follows the same "copy components into your app" ownership model as `shadcn/ui`
- it reduces compatibility risk compared with forcing React-focused `shadcn/ui` and Radix patterns through `preact/compat`
- it lets the project keep its current frontend runtime and test setup

The real objective is not merely to look like `shadcn/ui`. The objective is to move the app onto a stable, reusable component foundation that produces a coherent `shadcn`-like interface and is maintainable after the cutover.

## Design Principles

### 1. Preserve behavior, replace presentation

The data flow, backend semantics, and major interaction outcomes should stay the same. The rendering layer and UI composition model should change decisively.

### 2. Replace the shell, not just the paint

This project should not stop at swapping colors and rounded corners. The shell, panel structure, modal hierarchy, and mobile behavior should all be rebuilt around the new component system.

### 3. Project-owned primitives over opaque dependency magic

The `ui` layer should live inside the repository so the team can adjust behavior, styling, and accessibility details without fighting a third-party abstraction boundary.

### 4. Keep information density appropriate for a terminal companion

The final UI should feel more polished than the current frontend, but it should still read like a serious multi-pane tool rather than a sparse marketing app.

## UI Foundation

### Styling stack

Introduce this styling stack in `web/`:

- `Tailwind CSS 3`
- shared CSS custom properties for color, radius, and spacing tokens
- `shadcn-preact`-style primitives in `web/src/components/ui/`
- a much smaller global stylesheet for app-wide resets, background treatment, viewport behavior, and a small number of shell-level rules

The existing large CSS surfaces in `web/src/styles/global.css` and `web/src/styles/theme.css` should be reduced to foundation concerns instead of continuing to own every rendered surface.

### Token model

Adopt a token set compatible with `shadcn-preact`, including:

- `--background`
- `--foreground`
- `--card`
- `--card-foreground`
- `--popover`
- `--popover-foreground`
- `--primary`
- `--primary-foreground`
- `--secondary`
- `--secondary-foreground`
- `--muted`
- `--muted-foreground`
- `--accent`
- `--accent-foreground`
- `--destructive`
- `--destructive-foreground`
- `--border`
- `--input`
- `--ring`
- `--radius`

The visual direction should stay light-first and tool-oriented, with a more intentional background and panel hierarchy than the current flat treatment.

### Initial UI primitives

Create or import project-owned primitives under `web/src/components/ui/` for the components most needed by the current app:

- `button`
- `card`
- `input`
- `textarea`
- `dialog`
- `sheet`
- `popover`
- `badge`
- `separator`
- `tabs`
- `skeleton`
- `scroll-area`

Additional primitives can be added only when a rewritten surface truly needs them.

## Application Architecture After Cutover

The current high-level code boundaries remain valid and should stay in place:

- `web/src/app/` remains responsible for top-level wiring and shell composition
- `web/src/domains/*` remains responsible for polling, submission, state derivation, and API orchestration
- `web/src/components/*` remains responsible for feature-level rendering
- `web/src/components/ui/*` becomes the shared visual foundation
- `web/src/lib/*` remains the utility and API helper layer

The primary architectural change is that feature components stop defining their own full visual systems and start composing a shared primitive layer.

## Shell and Surface Redesign

### App shell

`web/src/app/AppShell.tsx` becomes the top-level composition point for a three-pane application shell:

- left column: sessions and launch controls
- center column: conversation timeline and composer
- right column: workspace tools and details

Desktop behavior:

- three-column layout remains the primary information architecture
- the center conversation column stays visually dominant
- panel boundaries should be clearer through card, separator, and muted surface treatment

Mobile behavior:

- conversation becomes the default primary surface
- sessions and workspace move into `sheet`-based off-canvas panels
- modal/backdrop and safe-area behavior should be explicit and reliable

### Sessions pane

`web/src/components/sessions/SessionsPane.tsx` and `web/src/components/sessions/SessionCard.tsx` should be rebuilt around card-style session rows.

Target treatment:

- header area with product title, backend status context, and primary new-session action
- session items rendered as cards rather than plain list surfaces
- clear active state, hover state, unread or busy signals, and concise metadata grouping
- better visual support for backend identity and session ownership without adding new semantics

### Conversation pane

`web/src/components/conversation/ConversationPane.tsx` should move to a structured message timeline built from reusable surfaces.

Target treatment:

- user and assistant content rendered as distinct message cards or bubbles
- tool, reasoning, event, and ask-user items rendered as secondary semantic cards within the same visual system
- sticky or semi-sticky conversation header for session context and quick actions if the current shell needs it
- better empty, loading, and incremental-update states via skeleton or muted placeholder surfaces

The existing message grouping, collapse rules, and rendering semantics should be preserved unless a visual change clearly improves readability without changing meaning.

### Composer

`web/src/components/composer/Composer.tsx` should become a shadcn-style input workbench.

Target treatment:

- textarea and controls wrapped in a deliberate card or toolbar shell
- send action visually prioritized
- queue or follow-up actions grouped cleanly
- mobile bottom anchoring and safe-area behavior preserved
- optional todo-related entry points styled as part of the same input system rather than separate ad hoc controls

### Workspace

`web/src/components/workspace/SessionWorkspace.tsx` and related workspace subviews should move from raw utility panes to structured tabs, cards, and sections.

Target treatment:

- tabs or sectional headers for files, queue, diagnostics, harness data, and related tool surfaces
- better readability for structured content through cards, labels, separators, and monospace data blocks where appropriate
- mobile presentation that does not force the workspace to permanently share screen real estate with the conversation

### New-session dialog

`web/src/components/new-session/NewSessionDialog.tsx` should be rebuilt on top of `dialog`, `input`, `tabs`, `badge`, and form-style field groupings.

Target treatment:

- strong modal hierarchy
- clear backend switching
- clearer grouping of path, session name, provider, model, reasoning, and worktree options
- preserved existing submission behavior and derived default logic

This dialog is one of the most stateful surfaces in the app and should keep its current business logic while receiving a full structural redesign.

## File Impact

Primary files expected to change:

- `web/package.json`
- `web/vite.config.ts`
- `web/tsconfig.json`
- `web/tsconfig.app.json`
- `web/src/app/AppShell.tsx`
- `web/src/components/sessions/SessionsPane.tsx`
- `web/src/components/sessions/SessionCard.tsx`
- `web/src/components/conversation/ConversationPane.tsx`
- `web/src/components/composer/Composer.tsx`
- `web/src/components/new-session/NewSessionDialog.tsx`
- `web/src/components/workspace/SessionWorkspace.tsx`
- `web/src/styles/global.css`
- `web/src/styles/theme.css`

Likely new files:

- `web/tailwind.config.*`
- `web/postcss.config.*`
- `web/src/components/ui/*`
- small feature-local presentational helpers where a large surface benefits from decomposition

## Dependency and Build Changes

The frontend will need the dependency and config changes implied by `shadcn-preact`.

Expected additions include:

- `tailwindcss@3`
- `postcss`
- `autoprefixer`
- `class-variance-authority`
- `clsx`
- `tailwind-merge`
- `tailwindcss-animate`

Additional dependencies should only be introduced when required by the specific imported primitives. The cutover should avoid pulling in the entire optional component universe if only a subset is used.

`vite.config.ts` and TypeScript path aliases should be updated so `@/` and `@ui/`-style imports work cleanly.

## Migration Strategy

Although the user-facing goal is a one-shot cutover, implementation should still proceed in this internal order:

### Stage 1: Foundation

- add Tailwind and PostCSS
- add theme tokens and base stylesheet changes
- add path aliases and base utility helpers
- import or adapt the initial `ui` primitive set

### Stage 2: Shell rewrite

- rebuild `AppShell`
- establish desktop and mobile panel behavior
- connect shell-level primitives such as sheets, separators, and layout cards

### Stage 3: Core surfaces

- rebuild sessions pane
- rebuild conversation pane
- rebuild composer
- rebuild workspace
- rebuild new-session dialog

### Stage 4: Cleanup

- remove now-dead legacy CSS blocks
- normalize feature-level class usage around the new primitives
- verify that no major surface still depends on old presentation assumptions

The final shipped result should feel like one cohesive rewrite even if the internal development order is staged.

## Risks

### Dependency integration risk

Adding Tailwind and `shadcn-preact` primitives changes the frontend build stack. Misconfigured aliases, content scanning, or CSS ordering can produce incomplete styling or broken builds.

### Cutover scope risk

A one-shot UI rewrite touches nearly every visible surface. The main failure mode is not build failure but subtle interaction regressions inside already-working flows.

### Stateful dialog regression risk

`NewSessionDialog` contains backend-aware defaults, resume handling, path validation, worktree settings, and submit state. It is visually easy to rewrite incorrectly if business logic gets entangled with new form markup.

### Conversation rendering risk

`ConversationPane` currently handles rich message kinds, grouping rules, and expandable content. A visual rewrite could unintentionally flatten semantics or break scroll behavior.

### Mobile shell risk

Moving to a more explicit `sheet`-driven mobile shell improves clarity, but it increases the need to verify safe-area, viewport height, overlay, and focus behavior on real mobile browsers.

## Validation Strategy

Validation should prioritize user-visible behavior and shell correctness over pixel-perfect styling first.

Required checks:

- frontend build succeeds in `web/`
- existing core component tests are updated and passing
- session list rendering and selection still work
- message rendering still covers the current message kinds
- composer submit flows still work
- new-session dialog still submits valid payloads for current backend choices
- workspace panels remain accessible and readable
- mobile sidebar and workspace sheets behave correctly

Testing emphasis:

- component tests for major surface interactions
- regression tests for `ConversationPane` and `NewSessionDialog`
- manual browser verification for desktop and mobile layouts

## Acceptance Criteria

The cutover is complete when:

- `web/` uses a `shadcn-preact`-style primitive layer for its main UI surfaces
- the main shell, sessions pane, conversation pane, composer, workspace, and new-session dialog are all rebuilt on that foundation
- the old large bespoke CSS is no longer the primary source of component styling
- the current behavior of session browsing, message viewing, sending, and new-session creation is preserved
- the interface reads as one coherent modern multi-pane tool on both desktop and mobile

## Out of Scope

The following are not part of this cutover:

- migrating the frontend to React
- redesigning backend APIs or store contracts
- introducing new product concepts unrelated to the current surfaces
- trying to import every available `shadcn-preact` component whether used or not
- exact pixel replication of the legacy static frontend

## Follow-On Work

After the cutover lands, later work can consider:

- extracting additional `ui` primitives as more surfaces mature
- optional dark mode if it serves the product rather than just mirroring the upstream demo
- deeper visual refinement for specialized panes such as files, diagnostics, and queue inspection
- expanding test coverage around mobile-specific interactions
