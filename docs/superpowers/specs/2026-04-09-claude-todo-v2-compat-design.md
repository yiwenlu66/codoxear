# Claude Todo V2 Compatibility Design

**Date:** 2026-04-09

## Goal

Make the web session todo experience compatible with Claude Todo V2 without breaking the existing Pi todo snapshot flow.

The compatibility target includes all current todo-related display surfaces:

- conversation timeline
- workspace/details
- composer todo surface
- future shared todo state handling between those surfaces

## Approved Direction

Use a backend-normalized compatibility layer as the primary integration point.

- Recognize Claude Todo V2 custom message events in the backend log parser
- Normalize those events into stable web-facing message fields
- Render recognized Claude Todo V2 task events in the main conversation timeline as dedicated activity cards
- Extend the shared todo snapshot item type so workspace/details and composer can accept future Claude Todo V2 task metadata without breaking existing Pi todo rendering
- Roll out in phases: first make Claude Todo V2 events visible and typed correctly, then add broader event-to-snapshot aggregation when more event samples are available

## User-Approved Scope

### In scope

- Support Claude Todo V2 task-assignment events from session logs
- Render Claude Todo V2 task-assignment events in the timeline
- Preserve existing Pi `todo_snapshot` behavior in workspace/details and composer
- Extend shared todo types for future Claude Todo V2 fields such as owner/assignment metadata
- Keep old Pi todo data and new Claude Todo V2 data compatible in one web UI
- Leave clear extension points for additional Claude Todo V2 event kinds later

### Out of scope for this first compatibility pass

- Reconstructing a full Claude Todo V2 task list from arbitrary event history
- Editing Claude Todo tasks in the browser
- Designing a new standalone todo API endpoint
- Replacing existing Pi todo snapshot generation
- Inventing a speculative full Claude Todo V2 state machine without more real event samples

## Why This Direction

The request is broader than a one-off card renderer. The same todo semantics need to work across the timeline, workspace/details, and composer surfaces.

A frontend-only patch would create conflicting sources of truth:

- timeline would reflect Claude Todo V2 events
- workspace/details and composer would still depend on legacy snapshot semantics
- later event types would force repeated UI-specific adaptation

A backend-normalized compatibility layer keeps the semantic meaning in one place and lets the frontend consume stable structures.

## Current Codebase Facts

The existing web todo flow already has a shared shape centered on `diagnostics.todo_snapshot`.

Relevant files:

- `codoxear/pi_messages.py` parses Pi logs and emits normalized events plus todo snapshots
- `web/src/lib/types.ts` defines `MessageEvent`, `TodoSnapshotItem`, and `TodoSnapshot`
- `web/src/components/conversation/ConversationPane.tsx` renders timeline events and currently supports `todo_snapshot` but not generic `custom_message`
- `web/src/components/composer/TodoComposerPanel.tsx` renders the composer-adjacent todo summary/list
- `web/src/components/workspace/TodoPopover.tsx` and `web/src/components/workspace/SessionWorkspace.tsx` render todo state from diagnostics

Observed behavior today:

- Existing todo display expects `available`, `error`, `progress_text`, and `items`
- Each todo item currently only uses `id`, `title`, `status`, and `description`
- Claude Todo V2 custom messages are not currently recognized or rendered as dedicated todo events

## Example Claude Todo V2 Input

A real sample from the user:

```json
{
  "type": "custom_message",
  "customType": "claude-todo-v2-task-assignment",
  "content": "Task #1 assigned to @Codex",
  "display": true,
  "details": {
    "taskId": "1",
    "taskListId": "3c7c0443-c037-47f4-8326-86c13e21403c",
    "subject": "Clarify Claude Todo V2 compatibility goal",
    "description": "Ask the user a focused requirement question about what Claude Todo V2 compatibility should mean in the web session todo UI and data model.",
    "owner": "Codex",
    "assignedBy": "team-lead",
    "timestamp": "2026-04-09T15:01:23.735Z"
  },
  "id": "fe60f4ff",
  "parentId": "46834883",
  "timestamp": "2026-04-09T15:02:02.525Z"
}
```

This proves at least one new compatibility requirement:

- a Claude Todo V2 event may arrive as `type: custom_message`
- the display-relevant semantics live in `customType`, `content`, and `details`
- assignment metadata belongs to the task, not just the plain text string

## Architecture

### 1. Backend event normalization in `codoxear/pi_messages.py`

Add a Claude Todo V2 recognition layer inside the Pi log normalization path.

Responsibilities:

- Detect entries where `type == "custom_message"`
- Read `customType`, `content`, `display`, `details`, and timestamps defensively
- Recognize known Claude Todo V2 custom types, starting with:
  - `claude-todo-v2-task-assignment`
- Normalize recognized events into a stable event object for the web message stream

Suggested normalized fields for recognized Claude Todo V2 messages:

```python
{
  "type": "custom_message",
  "custom_type": "claude-todo-v2-task-assignment",
  "text": "Task #1 assigned to @Codex",
  "display": True,
  "task_id": "1",
  "task_list_id": "3c7c0443-c037-47f4-8326-86c13e21403c",
  "subject": "Clarify Claude Todo V2 compatibility goal",
  "description": "Ask the user a focused requirement question...",
  "owner": "Codex",
  "assigned_by": "team-lead",
  "details": { ...original details... },
  "ts": 1775746922.525,
}
```

Guardrails:

- Preserve original `details` so future fields are not lost
- Keep unknown Claude Todo V2 event kinds as generic custom messages rather than dropping them
- Do not break existing `todo_snapshot`, `tool`, `tool_result`, or message extraction paths

### 2. Timeline rendering in `web/src/components/conversation/ConversationPane.tsx`

Extend the conversation timeline to support `custom_message` events.

Responsibilities:

- Add `custom_message` to the main timeline kinds
- Add an event label such as `Custom Event` or a Claude Todo-specific label when recognized
- Render `claude-todo-v2-task-assignment` as a dedicated task-assignment card instead of a plain generic system card

Suggested rendering for `claude-todo-v2-task-assignment`:

- title/body headline from `text`, e.g. `Task #1 assigned to @Codex`
- metadata row for:
  - `subject`
  - `owner`
  - `assigned_by`
- optional body text for `description`

Fallback behavior:

- Unknown `custom_message` types render through a generic custom/system card
- Missing fields fail soft and show whatever normalized text/details are available

This keeps the event visible immediately even before full snapshot aggregation exists.

### 3. Shared type expansion in `web/src/lib/types.ts`

Expand the frontend types without changing existing semantics.

`MessageEvent` additions:

- `custom_type?: string`
- `display?: boolean`
- `task_id?: string`
- `task_list_id?: string`
- `subject?: string`
- `description?: string`
- `owner?: string`
- `assigned_by?: string`

`TodoSnapshotItem` additions for forward compatibility:

- `owner?: string`
- `assigned_by?: string`
- `updated_at?: string`
- `source?: string`

These fields are additive only. Existing Pi todo UI should continue to work unchanged.

### 4. Workspace/details and composer compatibility strategy

For this first pass, workspace/details and composer remain snapshot-driven.

Meaning:

- `web/src/components/composer/TodoComposerPanel.tsx` continues to render from `diagnostics.todo_snapshot`
- `web/src/components/workspace/TodoPopover.tsx` and `web/src/components/workspace/SessionWorkspace.tsx` continue to use the same snapshot source
- The snapshot item type becomes Claude Todo V2-capable, but the backend does not yet fabricate a complete current-state list from task-assignment events alone

Why this boundary is intentional:

- one assignment event is not enough to reconstruct an authoritative task state model
- forcing event history into a guessed snapshot would create semantic bugs
- the user explicitly preferred compatibility over speculative overreach

### 5. Phased compatibility model

#### Phase 1: visible event compatibility

Deliver now:

- parse Claude Todo V2 custom messages
- render task-assignment events in timeline
- extend shared types for future compatibility

#### Phase 2: current-state snapshot compatibility

Deliver after more real Claude Todo V2 samples exist:

- recognize additional event kinds such as claim/complete/update/block/unassign
- define event ordering and replacement rules
- fold those events into a unified current todo snapshot consumable by workspace/details and composer

This phased model avoids guessing hidden invariants too early.

## Data Flow

### Phase 1

1. Session log contains a Claude Todo V2 custom message.
2. `codoxear/pi_messages.py` recognizes and normalizes it.
3. `/api/sessions/<id>/messages` returns the normalized `custom_message` event.
4. `ConversationPane` renders a Claude Todo V2 task-assignment card.
5. Existing diagnostics todo snapshot flow remains unchanged.

### Later Phase 2

1. Multiple Claude Todo V2 custom events are recognized.
2. Backend folds them into a current task-state model using explicit event semantics.
3. Diagnostics exposes a unified compatible snapshot.
4. Composer/workspace surfaces render Claude Todo V2 task state through the same snapshot contract.

## Rendering Rules

### Timeline: Claude Todo V2 task assignment

Render when:

- `type === "custom_message"`
- `custom_type === "claude-todo-v2-task-assignment"`
- event is marked displayable or otherwise not filtered out by backend policy

Display:

- primary text: assignment summary
- subject line: task subject when present
- metadata chips/rows for owner and assigned_by
- description block when present

### Timeline: unknown custom message

Render a generic event card with:

- best available title or text
- optional detail dump only if already normalized into safe fields
- no crash when fields are missing or unexpected

### Workspace/details and composer

No visual behavior change is required in phase 1.

The compatibility work there is structural:

- accept future snapshot item fields
- avoid rejecting Claude Todo V2-enriched snapshot items later
- preserve current Pi todo rendering now

## Error Handling

- Missing `details`: still render the message text if present
- Missing assignment metadata: render partial card instead of dropping event
- Unknown `customType`: keep as generic custom message
- Invalid or absent timestamps: use existing fallback timestamp logic
- `display == false`: preserve the flag in normalized output so policy can be enforced consistently later

## Testing Strategy

### Python tests

Add focused tests around Claude Todo V2 event normalization in the Pi parser.

Cover:

- recognized `claude-todo-v2-task-assignment` event normalizes expected fields
- original `details` object is preserved
- ISO timestamps are parsed correctly
- `display` is preserved
- unknown Claude Todo V2 custom message does not crash and still returns a generic normalized event

### Frontend timeline tests

Update `web/src/components/conversation/ConversationPane.test.tsx` to cover:

- task-assignment event renders headline, subject, owner, assigned_by, and description
- unknown `custom_message` still renders a safe fallback card
- legacy event rendering remains unchanged

### Type compatibility checks

Keep existing todo snapshot tests green while extending types.

Pay attention to:

- `web/src/components/composer/Composer.test.tsx`
- `web/src/components/workspace/TodoPopover.test.tsx`
- existing timeline tests for `todo_snapshot`

## Acceptance Criteria

1. A Claude Todo V2 task-assignment custom message appears in the web timeline as a dedicated task event.
2. The event includes normalized task metadata instead of only raw text.
3. Existing Pi `todo_snapshot` rendering in composer, workspace/details, and timeline does not regress.
4. Shared frontend types can represent future Claude Todo V2 task metadata without breaking old snapshots.
5. Unknown Claude Todo V2 custom messages fail soft instead of disappearing or crashing the UI.

## Risks and Guardrails

- Do not infer a complete task state machine from one observed assignment event.
- Do not create separate frontend-only todo semantics that drift from backend normalization.
- Do not break the current Pi todo snapshot path while expanding types.
- Do not drop original event detail payloads that future compatibility work may need.

## Implementation Notes

- Prefer adding a dedicated normalization helper in `codoxear/pi_messages.py` instead of scattering `custom_message` checks across unrelated parsing branches.
- Prefer additive TypeScript changes over replacing existing interfaces.
- Prefer a dedicated task-assignment renderer in `ConversationPane` over forcing the event into generic assistant/system styling.
- Defer full event-history-to-snapshot folding until more Claude Todo V2 samples define the real invariant.
