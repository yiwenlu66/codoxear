# UI Cleanup Design

**Date:** 2026-04-08

## Goal

Reshape the `web/` frontend so the page feels ordered, quiet, and readable instead of crowded, inconsistent, and over-framed.

The design target is a restrained, professional "editorial canvas" interface:

- left side for stable session navigation
- right side for the conversation canvas and top-level actions
- no permanently mounted third workspace column
- workspace details available on demand from a toolbar-triggered dialog

## Approved Direction

Use a two-part shell with an on-demand workspace dialog.

User-approved decisions:

- Keep a persistent left `Sessions` rail on desktop
- Make the right side the single main conversation canvas
- Remove the current always-visible workspace rail from the desktop shell
- Open `Workspace` from a button above the conversation as a dialog or lightweight panel
- Keep the visual tone restrained, quiet, and professional rather than flashy

## Problem Statement

The current UI feels messy for structural reasons, not just styling reasons.

Observed issues in the current implementation:

- Too many surfaces independently define borders, shadows, radii, and background treatments in `web/src/styles/global.css`
- The shell still carries a strong legacy three-column treatment even though the center conversation area should be the clear focal point
- Toolbar controls, session cards, and composer controls mix fixed-size and content-size behavior in ways that create awkward wrapping and overflow
- Long session titles and previews are not governed by one clear truncation or wrapping strategy
- The page hierarchy depends too much on repeated outlines instead of spacing, contrast, and consistent component sizing

## In Scope

- Redesign the desktop shell into a left rail + right canvas layout
- Convert workspace access from a persistent rail into a toolbar-triggered dialog
- Reduce border noise and unify surface hierarchy
- Standardize button sizing, spacing, and toolbar/composer rhythm
- Add explicit text overflow behavior for sessions, toolbar labels, and composer controls
- Keep the mobile experience consistent with the desktop mental model
- Reuse existing product behavior and session/workspace data flows wherever possible

## Out of Scope

- New backend endpoints or data models
- Reworking session semantics, queue behavior, or diagnostics semantics
- Redesigning the product into a radically branded or highly expressive visual system
- Turning the workspace into a permanently floating inspector with separate state semantics
- Feature additions unrelated to the current cleanup goal

## Why This Direction

This direction solves the user's complaints more directly than a lighter restyle.

- A two-part shell gives the page an obvious visual grammar: navigate on the left, work on the right
- Removing the permanent workspace rail gives width and attention back to the conversation
- Making workspace on-demand preserves utility without allowing it to dominate the shell
- A quieter surface system fixes the "weird borders everywhere" problem at the root instead of cosmetically repainting it
- Standardizing control rhythm addresses the overflowing and misshapen button issues systematically

## Information Architecture

### Desktop layout

The desktop shell becomes a two-zone layout.

#### 1. Sessions rail

The left rail remains visible and stable.

Responsibilities:

- session list and active-state navigation
- primary new-session action
- compact backend and ownership metadata
- lightweight supporting preview text

Constraints:

- visually lighter than the conversation canvas
- narrower and calmer than the current left column
- session rows behave like a disciplined list, not a stack of heavy feature cards

#### 2. Conversation canvas

The right side becomes the clear primary work surface.

Responsibilities:

- title and top-level toolbar
- conversation timeline
- composer
- entry point for workspace dialog and other utility actions

Constraints:

- brightest and most spacious surface in the shell
- the page's visual center of gravity
- conversation width should not be reduced by a permanently visible third rail

### Workspace behavior

The workspace no longer exists as a third permanent column.

New behavior:

- a toolbar button opens the workspace
- the workspace appears as a dialog or anchored floating panel over the conversation area
- closing the workspace fully returns width to the conversation canvas
- existing workspace content can be reused, but it is presented in a dialog container instead of a shell rail

Expected sections remain familiar:

- diagnostics
- queue
- files
- other existing detail panels already owned by the workspace UI

## Visual Design Principles

### 1. Make the conversation area the brightest surface

The right canvas should feel like the main page, not just one column among peers.

Implications:

- left rail uses calmer contrast and lighter emphasis
- workspace dialog becomes the strongest temporary surface only while open
- global page background stays quiet and supportive

### 2. Use fewer borders, with more intention

The current UI overuses borders and outlines.

New rule:

- borders are for actual interactive surfaces and overlays
- page regions should rely more on spacing, tonal separation, and consistent material treatment
- do not give every nested block a distinct heavy outline

### 3. Reduce surface vocabulary

The interface should use a small number of recognizable surface types.

Planned surface hierarchy:

- page background
- left navigation rail
- main conversation canvas
- lightweight cards within lists
- elevated dialogs and sheets

Each layer should have one consistent material treatment instead of many ad hoc variants.

### 4. Quiet professional tone

The visual language should stay restrained.

Implications:

- limited accent usage, concentrated on primary actions and active state
- no loud gradients or decorative color blocks in primary workflows
- typography and spacing should carry most of the hierarchy

## Component-Level Design

### Sessions pane

`web/src/components/sessions/SessionsPane.tsx` and `web/src/components/sessions/SessionCard.tsx` should move toward a calmer navigation list.

Design intent:

- lighter cards with clearer active state
- less ornamental framing
- stronger text discipline for long titles and preview copy
- metadata grouped compactly instead of scattered as competing visual elements

Rules:

- session title may wrap, but should stay bounded to a compact presentation
- preview text should clamp or wrap within a fixed rhythm and never expand the card unpredictably
- badges and state indicators should remain legible without over-emphasizing secondary metadata

### Conversation toolbar

The toolbar should feel quieter and more useful.

Design intent:

- title remains readable and central to the canvas identity
- utility actions are grouped and rhythmically sized
- workspace trigger is prominent enough to discover, but not louder than the conversation title

Rules:

- icon-only buttons stay square and fixed-size
- text buttons size to content with stable horizontal padding
- toolbar groups wrap gracefully on smaller widths without causing overlap or clipping

### Conversation timeline

The timeline should feel like a readable canvas instead of a pile of outlined blocks.

Design intent:

- user and assistant surfaces remain clearly differentiated
- tool and system surfaces still work, but they should not overpower ordinary conversation
- message cards use consistent spacing and surface logic

Rules:

- message widths stay bounded for readability
- long code, tables, and inline paths continue to wrap or scroll safely
- specialized message types keep semantic distinction without introducing a separate visual system for each type

### Composer

`web/src/components/composer/Composer.tsx` should become rhythmically stable.

Design intent:

- controls align to a consistent height system
- input remains the dominant control in the composer row
- attach, queue, and send actions should no longer distort the row when text length changes elsewhere

Rules:

- icon controls use fixed square hit areas
- send button keeps one stable size and shape
- text input container owns the available width
- no button text should overflow its container

### Workspace dialog

`SessionWorkspace` content should be reused inside a dialog-capable container.

Design intent:

- open from the conversation toolbar
- visually elevated above the canvas
- wide enough to read diagnostics and file lists comfortably
- easy to dismiss without feeling like a page transition

Rules:

- dialog should have bounded width and height with internal scrolling where needed
- overscroll should be contained
- backdrop and focus handling should remain accessible and predictable
- mobile may use the existing sheet pattern if that produces better ergonomics than a centered dialog

## Text Overflow and Sizing Rules

A major goal of this redesign is to eliminate uncontrolled overflow.

### Sessions

- session titles: bounded multi-line wrapping or clamping, not unlimited growth
- preview text: bounded to a compact preview area
- metadata rows: allowed to wrap, but with controlled spacing and `min-width: 0` behavior on flex children

### Toolbar

- icon-only controls remain fixed-size
- content-width controls keep stable padding and never shrink below their readable minimum
- title area should truncate or wrap intentionally rather than collide with action groups

### Composer

- input region owns remaining width with `min-width: 0`
- auxiliary buttons keep fixed dimensions
- queue or other secondary controls should not force the input off-screen

### Message surfaces

- keep existing safe handling for code blocks, tables, and long paths
- continue to ensure horizontal overflow is contained only where truly necessary
- preserve `overflow-wrap`/`break-word` style behavior for unpredictable content

## Responsive Behavior

### Desktop

- left rail remains visible
- right canvas fills the rest of the shell
- workspace opens as an overlay dialog or anchored panel from the toolbar

### Tablet and mobile

The mental model stays the same, but persistent chrome reduces.

Behavior:

- conversation remains the default visible surface
- sessions move into a left-side sheet or menu-triggered panel
- workspace remains on-demand rather than permanent
- composer stays pinned in a stable bottom region and must not jitter as controls wrap

This preserves the two-part logic conceptually even when the narrow viewport cannot show both zones at once.

## Accessibility and Interaction Notes

- Keep visible focus treatment for toolbar controls, session rows, dialog controls, and composer actions
- Maintain `aria-label` coverage for icon-only buttons
- Use proper dialog semantics and focus management for the workspace surface
- Ensure mobile sheets and dialogs contain overscroll appropriately
- Keep destructive actions and confirmation flows unchanged from current behavior

## Implementation Shape

### Primary files likely involved

- `web/src/app/AppShell.tsx`
- `web/src/components/sessions/SessionsPane.tsx`
- `web/src/components/sessions/SessionCard.tsx`
- `web/src/components/conversation/ConversationPane.tsx`
- `web/src/components/composer/Composer.tsx`
- `web/src/components/workspace/SessionWorkspace.tsx`
- `web/src/components/ui/dialog.tsx`
- `web/src/components/ui/sheet.tsx`
- `web/src/styles/global.css`

### Shell changes

`AppShell` should:

- stop rendering the persistent `workspaceRail` in desktop layout
- render only the left sessions rail and right conversation column in the main shell
- route workspace access through dialog state owned at shell level
- preserve existing mobile session sheet behavior where useful, but align it with the new two-part model

### Styling changes

`web/src/styles/global.css` should be reduced and reorganized around fewer shell-level rules.

Focus areas:

- shell grid and spacing
- surface hierarchy
- session card rhythm
- toolbar rhythm
- composer rhythm
- dialog sizing and overflow containment
- text truncation/wrapping behavior

The main goal is not to add more exceptions, but to remove contradictory legacy rules and replace them with a smaller, more coherent layout system.

## Testing Strategy

### Component and shell tests

Update existing tests to verify:

- the desktop shell no longer renders a permanent workspace rail
- workspace opens from a toolbar action in a dialog/sheet container
- sessions navigation remains functional
- conversation and composer still render correctly with the new shell

### Layout behavior checks

Add or update checks for:

- no unexpected horizontal overflow in shell-level containers
- long session titles remain bounded
- toolbar action group remains usable at narrower widths
- composer controls retain stable sizing

### Manual verification expectations

Before calling the redesign complete, verify:

- desktop layout feels visually calmer than the current shell
- workspace dialog is discoverable and comfortable to use
- mobile session access and workspace access still feel coherent
- long and mixed-language text does not produce broken buttons or crushed layouts

## Acceptance Criteria

The redesign is successful when all of the following are true:

- The desktop UI reads as two parts, not three competing columns
- The conversation canvas is the obvious focal surface
- Workspace details are available on demand without permanently consuming layout width
- Border and shadow noise are noticeably reduced
- Session cards, toolbar controls, and composer controls follow one consistent sizing rhythm
- Long text no longer causes obvious button overflow or layout breakage
- Mobile keeps the same mental model without persistent clutter

## Notes

The repository currently has many unrelated in-progress changes in the working tree.

To avoid mixing this design step with unrelated user work, this spec should be written and reviewed first. Committing it should only happen if it can be isolated safely from those existing changes.