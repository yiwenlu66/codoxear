# Session List Compact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert session sidebar items from large stacked cards into compact two-line list rows with inline metadata and hover-only icon actions.

**Architecture:** Keep the existing session grouping and store behavior unchanged, but refactor `SessionCard` markup so the main row content and actions live in a single compact surface. Then tighten the session-specific CSS in `global.css` so rows, badges, and group shells all read as a dense list while preserving active, hover, and focus states.

**Tech Stack:** Preact, TypeScript, Vitest, shared button/badge UI primitives, global CSS

---

## File Structure

### Files to modify
- `web/src/components/sessions/SessionCard.tsx` — replace the text-button action row with inline icon buttons, compress the content structure into a compact two-line row, and preserve selection/action semantics.
- `web/src/components/sessions/SessionsPane.test.tsx` — update coverage to assert compact-row structure, inline actions, and icon-button accessibility hooks.
- `web/src/styles/global.css` — reduce card/group visual weight, collapse multi-line content into denser rows, and add hover/focus styles for the icon action cluster.

### Files to leave untouched
- `web/src/components/sessions/SessionsPane.tsx` — no behavioral changes needed; it already wires actions and selection correctly.
- `web/src/components/sessions/SessionGroup.tsx` — keep group behavior unchanged; only tone down its styling from CSS.
- `web/src/components/ui/button.tsx` — existing `Button` variants are enough for compact ghost/icon controls.

## Task 1: Lock the compact-row DOM contract in tests

**Files:**
- Modify: `web/src/components/sessions/SessionsPane.test.tsx`
- Test: `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] **Step 1: Write the failing test expectations for inline action buttons and one-line metadata hooks**

Replace the existing "renders compact session rows with dense metadata hooks" test body with the following assertions so the current text-button layout fails immediately:

```tsx
  it("renders compact session rows with inline icon actions and compressed metadata", () => {
    renderSessionsPane({
      items: [
        {
          session_id: "sess-compact-1",
          alias: "Ship mobile polish",
          first_user_message: "Optimize phone UI",
          cwd: "/Users/huapeixuan/Documents/Code/codoxear/web",
          agent_backend: "pi",
          owned: true,
          queue_len: 3,
          busy: false,
        },
      ],
      activeSessionId: "sess-compact-1",
      loading: false,
      cwdGroups: {},
      newSessionDefaults: null,
    });

    const card = root?.querySelector("[data-testid='session-card']");
    expect(card?.querySelector(".sessionCardButton.compactSessionButton")).not.toBeNull();
    expect(card?.querySelector(".sessionCardMainRow")).not.toBeNull();
    expect(card?.querySelector(".sessionCardSecondaryRow")).not.toBeNull();
    expect(card?.querySelector(".sessionMetaBadges")).not.toBeNull();
    expect(card?.querySelector(".sessionActionRowInline")).not.toBeNull();

    const actionButtons = Array.from(card?.querySelectorAll<HTMLButtonElement>(".sessionActionIconButton") || []);
    expect(actionButtons.map((button) => button.getAttribute("aria-label"))).toEqual([
      "Edit session",
      "Duplicate session",
      "Delete session",
    ]);

    expect(card?.querySelector(".sessionPreviewInline")?.textContent).toContain("Optimize phone UI");
    expect(card?.querySelector(".sessionMetaText")?.textContent).toContain("sess-com");
  });
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
cd web && pnpm test -- SessionsPane.test.tsx
```

Expected: FAIL because `.sessionCardMainRow`, `.sessionCardSecondaryRow`, `.sessionActionRowInline`, and `.sessionActionIconButton` do not exist yet, and the preview/session id are still arranged in the old card layout.

- [ ] **Step 3: Add a second failing test that the text action row is gone**

Append this test near the compact-row assertions:

```tsx
  it("does not render legacy text action buttons inside session rows", () => {
    renderSessionsPane({
      items: [
        {
          session_id: "sess-actions-1",
          alias: "Refine sidebar density",
          first_user_message: "Make the cards smaller",
          cwd: "/work/codoxear/web",
          agent_backend: "codex",
          owned: false,
        },
      ],
      activeSessionId: "sess-actions-1",
      loading: false,
      cwdGroups: {},
      newSessionDefaults: null,
    });

    const cardText = root?.querySelector("[data-testid='session-card']")?.textContent || "";
    expect(cardText).not.toContain("Edit");
    expect(cardText).not.toContain("Duplicate");
    expect(cardText).not.toContain("Delete");
  });
```

- [ ] **Step 4: Run the same focused test command again**

Run:

```bash
cd web && pnpm test -- SessionsPane.test.tsx
```

Expected: FAIL again, now also confirming the legacy text action labels are still present in the rendered output.

- [ ] **Step 5: Commit the red test stage**

```bash
git add web/src/components/sessions/SessionsPane.test.tsx
git commit -m "test: define compact session row contract"
```

## Task 2: Refactor `SessionCard` into a compact row with inline icon actions

**Files:**
- Modify: `web/src/components/sessions/SessionCard.tsx`
- Test: `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] **Step 1: Replace the current card-heavy markup with compact-row structure**

Update `web/src/components/sessions/SessionCard.tsx` to the following shape. Keep the existing props and helper names, but replace the render body and add a tiny inline SVG icon helper so there is no new package dependency:

```tsx
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

import type { SessionSummary } from "../../lib/types";

interface SessionCardProps {
  session: SessionSummary;
  active: boolean;
  onSelect: () => void;
  onEdit?: () => void;
  onDuplicate?: () => void;
  onDelete?: () => void;
}

function shortSessionId(sessionId: string) {
  const match = sessionId.match(/^([0-9a-f]{8})[0-9a-f-]{20,}$/i);
  return match ? match[1] : sessionId.slice(0, 8);
}

function ActionIcon({ kind }: { kind: "edit" | "duplicate" | "delete" }) {
  if (kind === "edit") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M11.9 2.3a1.5 1.5 0 0 1 2.1 2.1l-7.4 7.4-3 .8.8-3 7.5-7.3Z" />
        <path d="m10.7 3.5 1.8 1.8" />
      </svg>
    );
  }
  if (kind === "duplicate") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <rect x="5" y="3" width="8" height="9" rx="1.5" />
        <path d="M3.5 6.5V12A1.5 1.5 0 0 0 5 13.5h5.5" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M3.5 4.5h9" />
      <path d="M6 4.5V3.4c0-.5.4-.9.9-.9h2.2c.5 0 .9.4.9.9v1.1" />
      <path d="m5 6.2.5 6c.1.7.6 1.3 1.4 1.3h2.2c.7 0 1.3-.6 1.4-1.3l.5-6" />
    </svg>
  );
}

export function useDesktopSessionActions() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return Boolean(window.matchMedia("(hover: hover) and (pointer: fine) and (min-width: 881px)").matches);
}

export function SessionCard({ session, active, onSelect, onEdit, onDuplicate, onDelete }: SessionCardProps) {
  const title = session.alias || session.first_user_message || session.title || shortSessionId(session.session_id);
  const preview = (session.alias ? session.first_user_message : session.cwd) || session.cwd || session.title || session.session_id;
  const desktopActions = useDesktopSessionActions();
  const hasActions = Boolean(onEdit || onDuplicate || onDelete);
  const showActions = hasActions && (active || desktopActions);

  return (
    <div
      data-testid="session-card"
      className={cn("sessionCard", active && "active")}
      aria-current={active ? "true" : undefined}
    >
      <Card className="sessionCardSurface">
        <CardContent className="sessionCardContent">
          <button
            type="button"
            className="sessionCardButton compactSessionButton"
            aria-current={active ? "true" : undefined}
            onClick={onSelect}
          >
            <div className="sessionCardMainRow">
              <div className="sessionTitleWrap">
                <div className="sessionTitle">{title}</div>
              </div>
              <div className="sessionCardHeaderAside">
                <div className="sessionMetaBadges">
                  <span className={cn("stateDot", session.busy && "busy")} />
                  <Badge variant="secondary" className="backendBadge">{session.agent_backend || "codex"}</Badge>
                  {session.owned ? <Badge variant="outline" className="ownerBadge">web</Badge> : null}
                  {session.queue_len ? <Badge className="queueBadge">{session.queue_len}</Badge> : null}
                </div>
                {showActions ? (
                  <div className="sessionActionRowInline">
                    {onEdit ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="sessionActionIconButton"
                        aria-label="Edit session"
                        onClick={(event) => {
                          event.stopPropagation();
                          onEdit();
                        }}
                      >
                        <ActionIcon kind="edit" />
                      </Button>
                    ) : null}
                    {onDuplicate ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="sessionActionIconButton"
                        aria-label="Duplicate session"
                        onClick={(event) => {
                          event.stopPropagation();
                          onDuplicate();
                        }}
                      >
                        <ActionIcon kind="duplicate" />
                      </Button>
                    ) : null}
                    {onDelete ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="sessionActionIconButton sessionActionIconButtonDanger"
                        aria-label="Delete session"
                        onClick={(event) => {
                          event.stopPropagation();
                          onDelete();
                        }}
                      >
                        <ActionIcon kind="delete" />
                      </Button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="sessionCardSecondaryRow">
              <div className="sessionPreview sessionPreviewInline">{preview}</div>
              <div className="sessionMetaLine">
                <span className="sessionMetaText">{shortSessionId(session.session_id)}</span>
              </div>
            </div>
          </button>
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 2: Run the focused test to verify the new DOM contract passes**

Run:

```bash
cd web && pnpm test -- SessionsPane.test.tsx
```

Expected: PASS for the new compact-row structure tests, while style-related visual differences are still pending.

- [ ] **Step 3: Sanity-check action behavior after the markup refactor**

Run:

```bash
cd web && pnpm test -- SessionsPane.test.tsx -t "selects grouped session cards when clicked"
```

Expected: PASS, confirming the main session button still selects rows even though action buttons now live inline and stop propagation.

- [ ] **Step 4: Commit the component refactor**

```bash
git add web/src/components/sessions/SessionCard.tsx web/src/components/sessions/SessionsPane.test.tsx
git commit -m "refactor: compact session row actions"
```

## Task 3: Compress the session sidebar styling to match the new row structure

**Files:**
- Modify: `web/src/styles/global.css`
- Test: `web/src/components/sessions/SessionsPane.test.tsx`

- [ ] **Step 1: Replace the oversized session/group CSS block with compact-row styles**

In `web/src/styles/global.css`, replace the existing session pane/group/card rules beginning at `.sessionsPane` through `.sessionMetaText` with the following compact styling block:

```css
.sessionsPane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 0 0.85rem 0.85rem;
  flex: 1;
}

.sessionsSurfaceHeader {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 0.9rem;
  padding: 0.25rem 0 0.8rem;
}

.sessionsEyebrow {
  margin: 0 0 0.2rem;
  color: var(--legacy-muted);
  font-size: 0.73rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.sessionsSurfaceTitle {
  margin: 0;
  font-size: 1.35rem;
  line-height: 1.05;
}

.sessionsSurfaceBody {
  min-height: 0;
  flex: 1 1 auto;
}

.sessionsList {
  display: flex;
  flex-direction: column;
  gap: 0.38rem;
  padding: 0.08rem 0.05rem 0.2rem;
}

.sessionGroupShell {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  padding: 0.38rem;
  border: 1px solid color-mix(in srgb, var(--border) 54%, transparent);
  border-radius: 1rem;
  background: color-mix(in srgb, var(--card) 90%, white 10%);
  box-shadow: 0 10px 24px -28px rgba(15, 23, 42, 0.5);
}

.sessionGroupHeader {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.65rem;
  width: 100%;
  padding: 0.18rem 0.2rem 0;
  border: 0;
  background: transparent;
  text-align: left;
  color: inherit;
}

.sessionGroupActions {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  flex: 0 0 auto;
}

.sessionGroupHeading {
  display: flex;
  flex-direction: column;
  gap: 0.12rem;
  min-width: 0;
}

.sessionGroupTitle {
  font-size: 0.92rem;
  font-weight: 640;
  line-height: 1.15;
}

.sessionGroupSubtitle {
  color: var(--legacy-muted);
  font-size: 0.77rem;
  line-height: 1.25;
  overflow-wrap: anywhere;
}

.sessionGroupList {
  display: flex;
  flex-direction: column;
  gap: 0.42rem;
}

.sessionCard {
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  text-align: left;
}

.sessionCardSurface {
  height: 100%;
  border: 1px solid color-mix(in srgb, var(--legacy-border) 28%, transparent);
  border-radius: 0.9rem;
  background: color-mix(in srgb, white 92%, var(--panel));
  box-shadow: none;
  transition: background-color 140ms ease, border-color 140ms ease;
}

.sessionCard:hover .sessionCardSurface,
.sessionCard:focus-within .sessionCardSurface {
  background: color-mix(in srgb, white 84%, var(--accent));
  border-color: color-mix(in srgb, var(--primary) 18%, var(--legacy-border));
}

.sessionCard.active .sessionCardSurface {
  border-color: color-mix(in srgb, var(--primary) 28%, var(--legacy-border));
  background: color-mix(in srgb, white 78%, var(--accent));
}

.sessionCardContent {
  position: relative;
  padding: 0;
}

.sessionCard[aria-current="true"] .sessionCardContent::before {
  content: "";
  position: absolute;
  left: 0;
  top: 8px;
  bottom: 8px;
  width: 2px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--legacy-accent) 88%, white);
}

.sessionCardButton {
  width: 100%;
  min-width: 0;
  padding: 0.55rem 0.7rem 0.52rem 0.78rem;
  border: 0;
  background: transparent;
  color: inherit;
}

.compactSessionButton {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.sessionCardMainRow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.55rem;
  min-width: 0;
}

.sessionTitleWrap {
  min-width: 0;
  flex: 1 1 auto;
}

.sessionCardHeaderAside {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.4rem;
  flex: 0 0 auto;
}

.sessionMetaBadges {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.28rem;
  flex-wrap: nowrap;
  flex: 0 0 auto;
}

.sessionActionRowInline {
  display: inline-flex;
  align-items: center;
  gap: 0.12rem;
  opacity: 0;
  transform: translateX(2px);
  transition: opacity 120ms ease, transform 120ms ease;
}

.sessionCard:hover .sessionActionRowInline,
.sessionCard:focus-within .sessionActionRowInline,
.sessionCard.active .sessionActionRowInline {
  opacity: 1;
  transform: translateX(0);
}

.sessionActionIconButton {
  width: 1.85rem;
  height: 1.85rem;
  min-width: 1.85rem;
  border-radius: 999px;
  color: var(--legacy-muted);
}

.sessionActionIconButton:hover,
.sessionActionIconButton:focus-visible {
  color: var(--text);
  background: color-mix(in srgb, var(--accent) 88%, white);
}

.sessionActionIconButtonDanger:hover,
.sessionActionIconButtonDanger:focus-visible {
  color: hsl(var(--destructive));
}

.sessionActionIconButton svg {
  width: 0.9rem;
  height: 0.9rem;
}

.sessionCardSecondaryRow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.55rem;
  min-width: 0;
}

.sessionPreview {
  color: var(--legacy-muted);
  font-size: 0.74rem;
  line-height: 1.25;
  overflow-wrap: anywhere;
}

.sessionPreviewInline {
  min-width: 0;
  flex: 1 1 auto;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sessionMetaLine {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  flex: 0 0 auto;
  min-width: 0;
  color: var(--legacy-muted);
}

.stateDot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: rgba(107, 114, 128, 0.85);
  box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.1);
}

.stateDot.busy {
  background: rgba(29, 78, 216, 0.95);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.backendBadge,
.ownerBadge,
.queueBadge {
  min-height: 1.2rem;
  padding: 0 0.38rem;
  border-radius: 999px;
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  text-transform: lowercase;
}

.backendBadge {
  background: rgba(15, 23, 42, 0.07);
}

.ownerBadge {
  background: rgba(255, 255, 255, 0.92);
  color: var(--legacy-muted);
}

.queueBadge {
  background: rgba(245, 158, 11, 0.9);
  color: #fff;
}

.sessionTitle {
  font-weight: 600;
  font-size: 0.9rem;
  line-height: 1.2;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sessionMetaText {
  font-size: 0.68rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

@media (max-width: 880px) {
  .sessionActionRowInline {
    opacity: 1;
    transform: none;
  }

  .sessionCardButton {
    padding-right: 0.62rem;
  }

  .sessionCardHeaderAside {
    gap: 0.28rem;
  }
}
```

- [ ] **Step 2: Run the focused component test after the CSS update**

Run:

```bash
cd web && pnpm test -- SessionsPane.test.tsx
```

Expected: PASS, confirming the renamed structure still matches and the CSS refactor did not require further DOM test changes.

- [ ] **Step 3: Run the broader web test suite for regression coverage**

Run:

```bash
cd web && pnpm test
```

Expected: PASS with no regressions in `SessionsPane`, dialog, composer, or shared UI tests.

- [ ] **Step 4: Build the web bundle to catch any TS/CSS integration issues**

Run:

```bash
cd web && pnpm build
```

Expected: PASS, producing the Vite build and copying assets into `codoxear/static/dist`.

- [ ] **Step 5: Commit the styling pass**

```bash
git add web/src/styles/global.css
git commit -m "style: compact session sidebar rows"
```

## Self-Review Checklist

- Spec coverage:
  - compact two-line list item: covered by Task 2 and Task 3
  - hover-only icon actions: covered by Task 1, Task 2, and Task 3
  - preserve backend/status/preview: covered by Task 2 and validated in Task 1 tests
  - tone down group shell styling: covered by Task 3
  - keep behavior/store logic unchanged: preserved by modifying only `SessionCard`, tests, and CSS
- Placeholder scan:
  - no `TODO`, `TBD`, or "implement later" placeholders remain
  - every code-edit step includes concrete code to paste or replace
  - every verification step includes exact commands and expected outcomes
- Type consistency:
  - class names referenced by tests (`sessionCardMainRow`, `sessionCardSecondaryRow`, `sessionActionRowInline`, `sessionActionIconButton`, `sessionPreviewInline`) are defined in the Task 2 markup and Task 3 CSS
  - action labels (`Edit session`, `Duplicate session`, `Delete session`) match the test expectations exactly

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-09-session-list-compact.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
