# Working Directory Session Groups

## Goal

Reorganize the web sidebar so sessions are grouped by `Working directory` instead of shown as one flat list. Each working directory becomes a collapsible group with a globally shared custom label for that absolute path.

## User Intent

The sidebar should reflect project structure rather than individual sessions alone:

- sessions from the same working directory should appear together
- each working directory should be expandable and collapsible
- each working directory should support its own user-defined name
- that name should be shared by all current and future sessions for the same absolute path

## Current State

The current sidebar renders a flat list of `SessionCard` items in `web/src/components/sessions/SessionsPane.tsx`.

Existing state and metadata are session-centric:

- session summaries already include `cwd`
- session aliases are editable per session
- sidebar metadata persists per session in `session_sidebar.json`
- recent cwd history persists separately in `recent_cwds.json`

There is no cwd-level metadata model, no cwd-level edit API, and no grouped sidebar presentation.

## Desired Behavior

### Grouping

- Group sessions by normalized absolute `cwd`.
- Sessions without a usable `cwd` go into a single fallback group labeled `No working directory`.
- Only groups that currently contain at least one session are shown.

### Group naming

- Each absolute working directory may have a custom label.
- The label is global for that path and reused by all current and future sessions under the same path.
- Clearing the label falls back to the directory basename.
- The UI should also show the real path in a secondary line when a real cwd exists, so similarly named directories remain distinguishable.

### Expand/collapse

- Each group has its own expand/collapse toggle.
- Collapse state persists across refreshes and across devices because it is stored server-side.
- Groups with active sessions are expanded by default unless the user explicitly collapses them.

### Session behavior inside a group

- Session cards keep their current behavior, actions, and visual language.
- Session ordering within a group should preserve the current backend-provided session ordering.
- Session-level alias editing remains separate from working-directory naming.

### Ordering

- Groups are sorted by the freshest activity among sessions in that group.
- The most recently active project directory should appear first.
- The fallback `No working directory` group follows the same freshness rule when present.

## Recommended Approach

Use a path-level cwd metadata store in the backend and keep `/api/sessions` session payloads flat.

This approach is preferred because:

- it preserves the existing session API shape with one additive field
- it keeps cwd semantics separate from session semantics
- it supports shared naming across clients and devices
- it avoids forcing the backend to emit a nested grouped response format
- it lets the frontend group sessions without changing existing session priority rules

## Data Model

Add a new persisted metadata file under the Codoxear app directory:

- `cwd_groups.json`

Suggested shape:

```json
{
  "/Users/huapeixuan/Documents/Code/codoxear": {
    "label": "Codoxear",
    "collapsed": false
  }
}
```

### Key rules

- The key is the normalized absolute cwd path.
- Entries are only created for real cwd-backed groups, not for the fallback group.
- `label` is optional or empty when unset.
- `collapsed` is a boolean and defaults to `false` when missing.

### Normalization rules

Backend normalization should:

- coerce input to string
- trim whitespace
- reject empty values for cwd-backed updates
- normalize to an absolute path using `Path(...).expanduser().resolve(strict=False)`
- store the normalized string form as the canonical key

This ensures that the same directory cannot accidentally receive multiple labels because of path formatting differences.

## API Design

### Extend `GET /api/sessions`

Keep returning the existing `sessions` array. Add an additive top-level field:

```json
{
  "sessions": [...],
  "cwd_groups": {
    "/Users/huapeixuan/Documents/Code/codoxear": {
      "label": "Codoxear",
      "collapsed": false
    }
  }
}
```

This keeps the API backward-friendly while giving the frontend enough information to group and render cwd sections.

### Add cwd-group edit endpoint

Add a small endpoint for cwd metadata updates, for example:

- `POST /api/cwd_groups/edit`

Request body:

```json
{
  "cwd": "/Users/huapeixuan/Documents/Code/codoxear",
  "label": "Codoxear",
  "collapsed": true
}
```

Rules:

- `cwd` is required for persisted cwd groups.
- `label` is optional; an empty string clears the custom name.
- `collapsed` is optional; if omitted, the stored collapse value is left unchanged.
- The response returns the normalized cwd key and the resulting metadata.

Example response:

```json
{
  "ok": true,
  "cwd": "/Users/huapeixuan/Documents/Code/codoxear",
  "label": "Codoxear",
  "collapsed": true
}
```

## Frontend Design

### Store updates

Update `web/src/lib/types.ts` and the sessions store so `/api/sessions` can carry cwd-group metadata.

Suggested client types:

- `CwdGroupMeta`
- `cwd_groups?: Record<string, CwdGroupMeta>` in `SessionsResponse`
- `cwdGroups` in the sessions store state

### Grouped rendering

In `web/src/components/sessions/SessionsPane.tsx`:

- stop rendering the session list as a single flat map
- derive grouped view data from the existing flat `items`
- keep session order stable within each group by iterating in the backend-provided order and pushing into buckets
- sort group buckets by the freshest session timestamp inside each bucket

Introduce a dedicated group component, for example:

- `web/src/components/sessions/SessionGroup.tsx`

Responsibilities:

- render group heading
- render expand/collapse affordance
- render inline rename affordance
- show secondary real-path text
- render child `SessionCard` items when expanded

### Group header content

Each group header should show:

- primary title: custom label if present, otherwise cwd basename
- secondary text: full normalized cwd path for real cwd groups
- expand/collapse chevron
- rename action

Fallback group behavior:

- title is fixed to `No working directory`
- no rename action
- no persisted cwd metadata
- no path subtitle

### Rename UX

Use inline rename in the group header.

Behavior:

- click rename or pencil icon to enter edit mode
- `Enter` saves
- `Escape` cancels
- blur saves
- saving an empty string clears the custom label and returns to basename display

This keeps cwd naming distinct from the existing session-level `Edit conversation` dialog.

### Collapse UX

- clicking the group header chevron toggles expansion
- the toggle posts the new `collapsed` value to the cwd-group endpoint
- optimistic UI is acceptable if failures restore the previous state and show a small inline error or reuse the pane-level error surface

## Backend Implementation Notes

In `codoxear/server.py` add cwd-group persistence parallel to the existing alias/sidebar metadata files.

Suggested responsibilities:

- load `cwd_groups.json` at startup
- validate that it is an object of objects
- sanitize values into `{label, collapsed}`
- expose an internal getter for the response payload
- expose an edit method that updates one cwd entry atomically and persists it
- omit empty entries where both label is empty and collapsed is false, so the file stays small

The session listing response should include both:

- existing `sessions`
- new `cwd_groups`

No backend regrouping is needed for this feature.

## Sorting Details

### Session ordering within a group

Do not invent a new per-group sort. Preserve the order already emitted by `SessionManager.list_sessions()`.

This preserves the current session-priority behavior, including any existing recency and priority-offset logic.

### Group ordering across groups

Compute each group's sort timestamp as the maximum relevant freshness value from its sessions.

Prefer:

- `updated_ts` when available
- otherwise a stable fallback such as start time if needed by the current payload

Then sort groups descending by that computed timestamp.

## Error Handling

- Invalid cwd-group update requests return `400` with a clear message.
- Unknown or empty cwd values are rejected for persisted cwd-group edits.
- If `cwd_groups.json` is malformed, log a warning, fall back to an empty map, and avoid crashing the server startup path.
- Frontend save failures should leave the last persisted state intact and show a visible error.

## Testing

### Backend tests

Add tests covering:

- loading valid and invalid `cwd_groups.json`
- cwd normalization collapsing equivalent path strings to one key
- clearing a label without losing collapse state
- saving collapse state updates without changing the label
- omission of empty default entries from persisted output
- `GET /api/sessions` returning `cwd_groups`
- cwd-group edit endpoint validating input and returning normalized output

### Frontend tests

Add tests covering:

- sessions render grouped by cwd
- two sessions with the same cwd appear under one group
- groups sort by the newest session activity
- group rename displays the custom label instead of basename
- empty rename reverts to basename
- collapse hides group contents and expand restores them
- fallback `No working directory` group renders for sessions without cwd
- existing session card actions still work inside groups

## Risks

- If path normalization is inconsistent between session discovery and edit requests, duplicate cwd entries could appear.
- If group sorting accidentally reorders sessions within buckets, existing sidebar priority behavior could regress.
- If rename and collapse use the same endpoint without partial-update discipline, one field could unintentionally overwrite the other.

## Out of Scope

This design does not include:

- empty cwd groups with no live sessions
- project favorites or manual reordering of groups
- moving sessions between groups
- replacing session aliases with cwd labels
- broader sidebar redesign beyond cwd grouping

## Acceptance Criteria

- The sidebar renders working-directory groups instead of one flat session list.
- All sessions with the same cwd appear under the same group.
- Each cwd group can be expanded or collapsed.
- Collapse state persists across page reloads and clients.
- Each cwd group can have a globally shared custom label keyed by absolute path.
- Clearing the custom label falls back to the directory basename.
- Session cards inside each group keep their current selection and action behavior.
- Group ordering reflects the freshest session in each group.
- Sessions without cwd appear in a fallback group.
