# Pi Session Todo Snapshot in Details

## Goal

Show the latest Pi todo list snapshot inside the existing web `Details` panel for a session, without adding a new top-level panel and without changing the chat message model.

## Background

Pi sessions already persist todo state in the session JSONL file when `manage_todo_list` runs successfully. In the current codebase:

- `codoxear/pi_messages.py` normalizes chat-visible events but drops non-subagent `toolResult` payload details.
- `codoxear/server.py` already exposes `/api/sessions/<id>/diagnostics`, which powers the existing `Details` modal in the browser.
- `codoxear/static/app.js` renders diagnostics as a current-state view, which matches the requested "latest snapshot" semantics.

The missing piece is a bounded server-side reader that can extract the newest todo snapshot from a Pi session file and expose it through diagnostics.

## User-Approved Scope

### In scope

- Show the latest todo snapshot for Pi-backed sessions inside `Details`
- Display progress summary plus todo items with status
- Preserve Pi-provided item order
- Show clear empty and unavailable states

### Out of scope

- Editing todo items from the web UI
- Showing historical todo versions
- Injecting todo updates into the chat transcript
- Adding a dedicated Todo modal or top-level button
- Reordering, filtering, or searching todo items

## Recommended Architecture

Use on-demand diagnostics enrichment.

When the browser opens `Details`, `codoxear/server.py` should enrich the existing diagnostics response with a `todo_snapshot` object for Pi sessions only. The server should compute this object by scanning the Pi session JSONL file from newest to oldest until it finds the most recent successful `manage_todo_list` result containing `details.todos`.

This keeps todo extraction out of the chat event pipeline and preserves the current meaning of the `Details` panel as a live snapshot instead of a historical timeline.

## Component Design

### 1. Pi session parsing helper

Add a dedicated helper in `codoxear/pi_messages.py` because parsing Pi session JSONL is already centralized there.

Suggested responsibility:

- Accept `session_path: Path`
- Read the file in bounded tail chunks using the existing JSONL helpers
- Search newest-first for the latest successful `manage_todo_list` tool result
- Accept any successful result with `toolName == "manage_todo_list"` and `details.todos` as an array, regardless of whether the tool call was `read` or `write`
- Normalize the todo payload into a compact snapshot structure for the server
- Return `None` when no valid snapshot exists
- Raise only on true file-read failures; malformed entries should be ignored

Suggested normalized shape:

```python
{
    "items": [
        {
            "id": 1,
            "title": "Explore project context",
            "description": "Inspect relevant files before changing behavior.",
            "status": "completed",
        },
    ],
    "counts": {
        "total": 4,
        "completed": 2,
        "in_progress": 1,
        "not_started": 1,
    },
    "progress_text": "2/4 completed",
}
```

Normalization rules:

- Preserve the original item order from Pi
- Preserve `id`, `title`, `description`, and `status`
- Map Pi status strings to API keys only for aggregate counts:
  - `in-progress` -> `in_progress`
  - `not-started` -> `not_started`
- Unknown statuses should not crash parsing; keep the raw item and exclude it from known counts if necessary

### 2. Diagnostics API extension

Extend `/api/sessions/<id>/diagnostics` in `codoxear/server.py`.

Behavior:

- For non-Pi sessions, return a todo payload that indicates no snapshot is available
- For Pi sessions with a readable `session_path`, call the new helper and attach its result
- Do not change any existing diagnostics fields or semantics
- Keep todo lookup best-effort so the rest of `Details` still renders even if todo extraction fails

Suggested response field:

```python
"todo_snapshot": {
    "available": True,
    "error": False,
    "items": [...],
    "counts": {
        "total": 4,
        "completed": 2,
        "in_progress": 1,
        "not_started": 1,
    },
    "progress_text": "2/4 completed",
}
```

Fallbacks:

- No todo found: `{"available": False, "error": False, "items": []}`
- Read failure: `{"available": False, "error": True, "items": []}`

This shape lets the UI distinguish between "no todo yet" and "todo unavailable" without failing the entire modal.

### 3. Details UI rendering

Update `codoxear/static/app.js` so the existing `showDiagViewer()` flow renders a new `Todo list` block when diagnostics include `todo_snapshot`.

Rendering rules:

- Place the block after the core session metadata rows
- Render a section title: `Todo list`
- If `available` is true:
  - Show the `progress_text` summary first
  - Render each todo item as a compact row/card
  - Main line shows `title`
  - Secondary line shows `description` only when present
  - Status is shown as a compact label using the raw Pi status value (`completed`, `in-progress`, `not-started`)
- If `available` is false and `error` is false, show `No todo list yet`
- If `available` is false and `error` is true, show `Todo list unavailable`

UI constraints:

- Reuse the current `Details` modal instead of adding a new modal
- Keep the styling visually lighter than the main chat bubbles
- Avoid introducing a complex color system in v1; subtle status chips are enough
- Keep the layout mobile-safe inside the existing diagnostics viewer width

### 4. CSS support

Add minimal new styles in `codoxear/static/app.css` for the todo block.

Suggested additions:

- Section wrapper inside the diagnostics modal
- Todo item list with vertical spacing
- Todo title/description text hierarchy
- Small status chip treatment for the three known states

The new styles should align with the current `Details` modal visual language rather than introducing a new panel design.

## Data Flow

1. User opens `Details` for a selected session in the browser.
2. `codoxear/static/app.js` requests `/api/sessions/<id>/diagnostics`.
3. `codoxear/server.py` builds the existing diagnostics payload.
4. If the session backend is `pi`, the server asks `codoxear/pi_messages.py` for the latest todo snapshot from `session_path`.
5. The server returns the diagnostics payload with `todo_snapshot` included.
6. The browser renders the normal diagnostics rows and then the todo block.

## Error Handling

### Missing or empty todo state

If the Pi session file exists but no valid `manage_todo_list` snapshot is found, return a clean empty-state payload. This should not be treated as an error.

### Invalid or partial tool payloads

If a matching tool result is malformed, skip it and continue searching older entries until a valid snapshot is found or the scan budget is exhausted.

### File access failures

If the file cannot be read, diagnostics should still succeed. Only the todo portion should degrade to `error = True`.

## Performance

Use bounded tail scanning instead of full-file parsing.

Guidelines:

- Start from the file tail because the newest snapshot is the only one needed
- Stop immediately after the first valid match
- Reuse existing JSONL read helpers where possible
- Keep the scan budget modest so opening `Details` remains cheap even for long Pi sessions

Because this happens only when `Details` is opened, the design deliberately prefers simplicity and correctness over background caching in v1.

## Testing Strategy

### Backend tests

Add focused coverage for the parsing helper and diagnostics response:

- Latest valid todo snapshot is returned from a Pi session file
- Newer malformed todo entries are skipped in favor of the next valid older snapshot
- No todo snapshot returns the empty state
- File-read failure returns the unavailable state without breaking diagnostics
- Non-Pi diagnostics keep working and report no todo snapshot

### Frontend verification

Verify the `Details` modal in these states:

- Pi session with todo snapshot
- Pi session with no todo yet
- Pi session with todo unavailable
- Non-Pi session

### Regression checks

- Existing diagnostics rows still render correctly
- Existing queue/file/harness UI remains unchanged
- No changes to chat event rendering behavior

## Why This Design

This design matches the requested UX with the smallest invariant-preserving change:

- It uses the existing `Details` panel instead of inventing a second surface
- It keeps todo extraction out of the chat pipeline, which avoids semantic confusion
- It preserves the meaning of `Details` as the latest session state
- It limits risk by touching only diagnostics-specific server and UI paths plus a focused Pi parsing helper
