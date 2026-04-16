# Legacy Web Parity Restoration

## Goal

Make the new `web/` frontend feel as close as practical to the legacy web UI from commit `c1e1c24`, especially for layout, visual hierarchy, and the main interaction flows that currently make the new page feel noticeably simpler.

The target is not to revive the legacy implementation. The target is to restore legacy user-facing semantics while keeping the current component and store architecture in `web/src/*`.

## Legacy Baseline

Use these legacy sources as the parity baseline:

- `c1e1c24:codoxear/static/app.css`
- `c1e1c24:codoxear/static/app.js`
- `c1e1c24:codoxear/static/index.html`

These files define the legacy shell layout, session list density, conversation styling, composer structure, mobile sidebar behavior, modal presentation, and the richer workspace/tooling surfaces that the current `web/` rewrite has not yet fully restored.

## Current Gap

The new frontend already has the right broad product surfaces:

- app shell
- session list
- conversation view
- composer
- new-session dialog
- session workspace

But the current implementation is still much flatter and more skeletal than the legacy UI:

1. `web/src/app/AppShell.tsx` presents a simplified shell without the legacy layering and mobile shell behavior.
2. `web/src/components/sessions/SessionsPane.tsx` renders a basic selectable list instead of information-dense session cards.
3. `web/src/components/conversation/ConversationPane.tsx` renders messages mostly as raw text blocks rather than legacy-like chat bubbles and grouped message surfaces.
4. `web/src/components/composer/Composer.tsx` keeps core send behavior but not the legacy visual treatment and shell integration.
5. `web/src/components/new-session/NewSessionDialog.tsx` is functionally present but visually much simpler than the legacy modal.
6. `web/src/components/workspace/SessionWorkspace.tsx` still exposes raw JSON-style diagnostics, queue, files, and request data rather than a proper legacy-like workspace panel.

## Chosen Direction

Rebuild the new frontend around legacy page semantics, but keep the current `web/src` architecture.

This means:

- keep `preact` components and stores
- keep the current API boundaries and polling model
- restore the legacy shell structure and visual language
- restore the highest-value legacy interactions first
- avoid directly transplanting the large imperative DOM logic from legacy `app.js`

This is a parity restoration project, not a redesign and not a rollback.

## Restoration Principles

### 1. User-visible parity over code-level imitation

If the user can perceive the legacy behavior, reproduce it. If a behavior only exists because of the old DOM implementation, do not copy the implementation itself.

### 2. Restore semantics before details

Rebuild the shell, hierarchy, and state presentation first. After that, fill in finer interactions.

### 3. Preserve data boundaries

Do not collapse the existing store/component boundaries merely to chase pixel parity. The UI should look and behave like the legacy version while remaining maintainable.

### 4. Prioritize the "old UI is back" feeling

The first implementation goal is to eliminate the impression that the new page is stripped-down or temporary.

## Scope Tiers

### Tier 1: Must restore early

These items determine whether the new UI immediately feels like the legacy UI:

- desktop shell structure with clear sidebar, conversation, and workspace columns
- mobile sidebar open/close behavior with backdrop handling
- legacy-like session card density, selected state, and core status indicators
- legacy-like conversation bubble treatment and message area spacing
- legacy-like composer shell, controls, spacing, and mobile safe-area behavior
- legacy-like new-session modal hierarchy and visual grouping
- workspace panel upgraded from raw JSON dumps to structured sections for diagnostics, queue, files, and requests

### Tier 2: Restore after shell parity

These items matter, but should follow after the shell and primary interactions are back:

- session-row hover action affordances
- selected quick actions and inline controls from the legacy session list
- richer picker and dialog styling details
- deeper queue/file viewer styling parity
- selected swipe or touch affordances where they fit the component model cleanly

### Out of scope for the first parity pass

- a literal 100% port of every imperative gesture from legacy `app.js`
- backend API changes
- major product behavior changes
- introducing new product concepts or redesign directions

## Proposed UI Architecture

### App shell

`web/src/app/AppShell.tsx` becomes the layout orchestrator for legacy-like shell behavior.

Responsibilities:

- restore the legacy app-shell proportions and layering
- manage mobile sidebar visibility and backdrop state
- place sidebar, conversation pane, composer, and workspace in a legacy-like relationship
- handle the conditions under which workspace is shown, stacked, or visually secondary

The shell should reflect the legacy `.app` / `.sidebar` / main-pane mental model rather than the current simplified grid.

### Sessions pane

`web/src/components/sessions/SessionsPane.tsx` will be rebuilt around session-card semantics instead of plain list buttons.

Responsibilities:

- restore a title/header area consistent with the legacy sidebar
- render information-dense session cards with room for title, metadata, and status markers
- expose selection state in a way that visually matches the legacy active card treatment
- reserve structure for later inline actions without overcomplicating the first pass

The first pass should make the left column feel recognizably like the legacy sidebar even before every secondary action is restored.

### Conversation pane

`web/src/components/conversation/ConversationPane.tsx` will move from raw event dumps to structured message rendering.

Responsibilities:

- render legacy-like message cards or bubbles for user and assistant turns
- distinguish `ask_user`, tool-related, and other non-standard messages in a readable way
- restore the spacing, scroll feel, and visual hierarchy of the legacy chat surface
- improve empty and loading states so they belong to the same visual system

### Composer

`web/src/components/composer/Composer.tsx` keeps the existing submit behavior but adopts the legacy input shell.

Responsibilities:

- restore the legacy composer container and control arrangement
- preserve keyboard and submit semantics already present in the store
- support multiline behavior without drifting away from the legacy visual treatment
- restore mobile safe-area and bottom anchoring behavior

### New-session dialog

`web/src/components/new-session/NewSessionDialog.tsx` will be visually reorganized to match the legacy modal hierarchy more closely.

Responsibilities:

- present backend choice, working directory, and start action in a clearer grouped layout
- restore a legacy-like modal feel rather than a basic form card
- preserve current functional fields while leaving room for richer provider/model/reasoning controls later

### Workspace panel

`web/src/components/workspace/SessionWorkspace.tsx` is the highest-priority structural cleanup after the shell.

Responsibilities:

- stop presenting diagnostics, queue, files, and requests mainly as raw JSON
- render each area as a deliberate section with headings and readable structure
- make the workspace feel like a proper legacy-adjacent tool panel
- stage deeper viewer parity later if needed

The first pass should prioritize readability and panel structure over reproducing every old viewer detail.

## File-Level Plan

Primary files to refactor:

- `web/src/app/AppShell.tsx`
- `web/src/components/sessions/SessionsPane.tsx`
- `web/src/components/conversation/ConversationPane.tsx`
- `web/src/components/composer/Composer.tsx`
- `web/src/components/new-session/NewSessionDialog.tsx`
- `web/src/components/workspace/SessionWorkspace.tsx`
- `web/src/styles/theme.css`
- `web/src/styles/global.css`

Possible supporting additions are allowed for focused layout helpers, small presentational subcomponents, or local utilities, but the work should avoid exploding into a large new subsystem.

## Styling Strategy

Use the legacy stylesheet as the design source, but translate it into the new frontend rather than copying it wholesale.

### Token layer

Reintroduce the legacy visual language through shared tokens such as:

- background and panel colors
- border and shadow rules
- selected-state colors
- sidebar width and responsive thresholds
- message bubble colors
- composer control sizing

### Global shell rules

Recreate the legacy shell behavior in `web/src/styles/global.css`:

- fixed-height app shell behavior
- column layout and responsive breakpoints
- mobile sidebar slide-in behavior
- backdrop and modal layering
- conversation and workspace overflow handling

### Component styling

Restore legacy-like treatment for:

- session cards
- message bubbles
- composer form shell
- modal cards
- workspace sections and controls

The translation should preserve current maintainability, so common shell rules and tokens should live in shared styles rather than being duplicated inside each component.

## Data and State Boundaries

The parity work should not redefine core frontend ownership.

Keep these existing boundaries intact:

- session list and selection state stay in the sessions store
- conversation loading and polling stay in the messages store
- workspace data loading stays in the session-ui store
- composer draft and send flow stay in the composer store

If a legacy-like display needs extra derived values, prefer:

1. component-level formatting for small display-only needs
2. shared selectors/helpers for repeated derived display logic
3. store changes only when current state truly cannot represent the needed UI

This keeps the project from regressing into legacy-style hidden coupling.

## Implementation Order

### Phase 1: Shell parity

- rebuild app shell layout
- restore sidebar shell and session-card structure
- restore conversation surface and composer shell
- restore mobile sidebar behavior

This phase should already remove most of the "too simple" feeling.

### Phase 2: Modal and workspace parity

- upgrade the new-session dialog structure and styling
- convert workspace JSON sections into structured UI sections
- align panel hierarchy with the legacy product feel

### Phase 3: Secondary interaction polish

- add selected inline actions, hover affordances, and tighter card details
- add selected mobile/touch behaviors where they fit the component model safely
- refine viewer/picker details if still needed after the main parity pass

## Risks

### Risk 1: Copying legacy DOM behavior too literally

The biggest architectural risk is rebuilding the same maintenance problem that existed in `codoxear/static/app.js`.

Mitigation:

- re-express behavior as component structure and local state
- keep stores as the source of async and polling behavior
- avoid global DOM manipulation patterns unless strictly necessary

### Risk 2: Workspace complexity

`SessionWorkspace` currently has the largest visual gap relative to legacy behavior and also the highest chance of scope expansion.

Mitigation:

- first restore readable panel structure
- postpone deeper viewer parity until the shell and primary interactions are solid

### Risk 3: Chasing total gesture parity too early

Legacy touch and swipe details can consume a lot of time while adding little to the immediate perception of parity.

Mitigation:

- restore only the most visible and structurally compatible gestures in the first pass
- defer exotic or brittle touch behavior until after the main parity pass is evaluated

## Validation Strategy

Validation is based on parity against the legacy sources, not on independent redesign taste.

### Primary comparison set

Compare the new implementation against the legacy baseline for these scenes:

- desktop app shell
- mobile app shell
- session list and selection states
- conversation area and message bubbles
- composer in idle and multiline states
- new-session modal
- workspace with diagnostics, queue, files, and requests visible

### Functional verification

At minimum, verify:

- session list refresh and selection
- initial message loading and polling
- composer send flow
- new-session creation flow
- workspace data refresh and request submission
- mobile sidebar usability

### Test posture

Update existing component tests where structure changes invalidate snapshots or expectations.
Add targeted render/interaction tests for the most important restored behaviors rather than relying only on visual inspection.

## Acceptance Criteria

The parity restoration is successful when:

- the new `web/` frontend is immediately recognizable as the legacy Codoxear UI in shell, hierarchy, and primary interaction feel
- the page no longer looks like a simplified placeholder compared with the legacy UI
- the implementation still uses the current component/store architecture rather than reverting to a monolithic imperative frontend
- desktop and mobile behavior both feel meaningfully closer to the legacy baseline
- the workspace panel is readable and structured instead of raw JSON-heavy

## Notes

- Per user instruction, this spec is written but not committed because the current git worktree already contains unrelated local changes.
- After user review and approval of this spec, the next step is to create an implementation plan and then execute the parity restoration in phases.
