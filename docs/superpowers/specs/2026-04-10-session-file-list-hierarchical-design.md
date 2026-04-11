# Session File List Hierarchical Design

Date: 2026-04-10

## Goal

Replace the current all-files-at-once `/api/sessions/:id/file/list` response with directory-by-directory querying so the workspace file browser stays fast and bounded even for large repositories.

## User Intent

The current file list endpoint returns too much data for large working directories.

The user wants:
- the existing `/api/sessions/:id/file/list` route to remain the canonical file-list endpoint
- the endpoint to return only one directory level at a time
- the browser UI to query and expand directories incrementally
- filtering to respect the session cwd root `.gitignore`
- no compatibility mode that keeps the old flat recursive payload

## Current State

The current implementation in `codoxear/server.py` uses `_list_session_relative_files(base)` to recursively walk the entire session cwd and returns a flat `files: string[]` payload from `/api/sessions/:id/file/list`.

The current frontend assumes that flat shape:
- `web/src/lib/types.ts` models `SessionFileListResponse` as `{ files: string[] }`
- `web/src/lib/api.ts` fetches the whole file list in one call
- `web/src/domains/session-ui/store.ts` preloads the file list during session refresh
- `web/src/components/workspace/FileViewerDialog.tsx` merges the flat list into the file chooser UI

This means opening the file viewer or refreshing session UI can trigger an expensive full-repo traversal and transfer a large payload that the user may never actually browse.

## Recommended Direction

Redefine `/api/sessions/:id/file/list` as a direct-children listing API with an optional `path` query parameter.

This is the simplest design that directly matches the user request. It keeps the route stable, gives the frontend a clean lazy-loading model, and prevents large repositories from forcing a full recursive scan up front.

## Design Overview

### 1. API Contract

Keep the route:
- `GET /api/sessions/:id/file/list`

Add query parameter:
- `path` optional; empty or missing means the session cwd root

Replace the response shape with a directory view:

```json
{
  "ok": true,
  "cwd": "/repo",
  "path": "src",
  "entries": [
    { "name": "components", "path": "src/components", "kind": "dir" },
    { "name": "main.tsx", "path": "src/main.tsx", "kind": "file" }
  ]
}
```

Entry rules:
- `name`: basename only
- `path`: session-relative normalized path using `/`
- `kind`: `dir` or `file`

Behavior:
- `path=""` returns direct children of the session cwd root
- `path="src"` returns direct children of `src`
- the endpoint never returns a recursive subtree

## 2. Path Resolution and Validation

Reuse the existing session path safety model:
- resolve from the session cwd
- reject path escape attempts outside the session cwd
- require the resolved target to exist
- require the target to be a directory

Error behavior:
- unknown session -> `404`
- `path` not found -> `404`
- `path` is not a directory -> `400`
- invalid relative path / escape attempt -> `400`
- permission failures -> `403`

### 3. Ignore Rules

Directory enumeration should apply two layers of filtering.

Built-in ignored directories remain in effect:
- `.git`
- `.hg`
- `.mypy_cache`
- `.pytest_cache`
- `.svn`
- `__pycache__`
- `build`
- `dist`
- `node_modules`
- `venv`
- `.venv`

In addition, the server should read only the root `.gitignore` at `session cwd/.gitignore`.

Rules for `.gitignore` handling:
- patterns are interpreted relative to the session cwd root
- only the root `.gitignore` is consulted
- nested `.gitignore` files are out of scope for this change
- ignored files are omitted from `entries`
- ignored directories are omitted from `entries` and cannot be expanded further
- if `.gitignore` is absent, enumeration falls back to the built-in ignore set only
- if `.gitignore` parsing fails, enumeration should degrade gracefully to built-in ignore rules instead of failing the endpoint

### 4. Ordering

Return entries in a stable, predictable order:
- directories first
- files second
- alphabetical by `name` within each group

This keeps the browser behavior familiar and avoids tree jitter between loads.

### 5. Backend Structure Changes

Expected backend refactor in `codoxear/server.py`:
- replace `_list_session_relative_files(base)` with a helper that lists one directory level at a time
- add a helper for loading and applying root `.gitignore` rules against session-relative paths
- have `/api/sessions/:id/file/list` parse `path`, resolve it safely, and return `entries`

The new backend helper should operate on a target directory but still evaluate ignore rules relative to the session cwd root.

### 6. Frontend Data Model Changes

Update the frontend contract so `SessionFileListResponse` reflects the new payload:
- remove the flat `files: string[]`
- add `path?: string`
- add `entries: Array<{ name: string; path: string; kind: "dir" | "file" }>`

`web/src/lib/api.ts` should continue exposing `getFiles`, but it now accepts an optional directory path argument and forwards it as a query parameter.

### 7. File Browser Interaction Model

The workspace file browser should become a lazy directory tree.

Behavior:
- opening the file viewer loads only the root directory
- expanding a directory triggers `/file/list?path=<dir>`
- collapsing a directory does not discard cached children immediately
- clicking a file still opens content through the existing read/diff endpoints
- `initialPath` continues to support directly opening a file even if its parent directories were not manually expanded first

Recommended ownership:
- keep directory-tree state local to `web/src/components/workspace/FileViewerDialog.tsx`
- do not keep the full browsed tree in `web/src/domains/session-ui/store.ts`

This keeps the heavier, interaction-local browsing state out of the shared session refresh path.

### 8. Frontend Node State

Each directory node should track enough state to support partial loading and partial failure:
- `path`
- `name`
- `kind`
- `loaded`
- `loading`
- `error`
- `children`

State rules:
- loading one directory should not block browsing elsewhere
- a failed child directory load should show retry UI for that node only
- cached children may be reused until the dialog closes, the session changes, or the user explicitly refreshes

### 9. Session UI Refresh Behavior

The current session refresh path should stop preloading the entire file list.

Recommended change:
- remove the eager `api.getFiles(sessionId)` call from `web/src/domains/session-ui/store.ts`
- stop treating the file browser data as a global session snapshot concern
- let the file viewer fetch directory listings on demand

This prevents switching sessions from paying the cost of a file-tree request that the user may not need.

### 10. Interaction with Direct File Access

This change is about file listing visibility, not necessarily read authorization semantics.

For this scope:
- `.gitignore` controls whether an entry appears in the browser tree
- existing file read endpoints may keep their current behavior unless implementation work later decides to align read access with listing visibility

This avoids coupling the file-tree redesign to a broader file-access policy change.

### 11. Testing / Verification Plan

Backend tests:
- root listing returns only direct children
- nested listing returns only that directory's direct children
- ordering is directories first, then files, alphabetized within each group
- built-in ignored directories remain excluded
- root `.gitignore` ignored files are excluded
- root `.gitignore` ignored directories are excluded
- invalid `path`, missing `path`, non-directory `path`, and escaped paths return the expected errors
- malformed or unreadable `.gitignore` degrades gracefully without breaking the endpoint

Frontend tests:
- opening the dialog requests only the root listing
- expanding a directory requests `/file/list?path=<dir>`
- re-expanding a cached directory does not refetch unnecessarily
- file clicks still open file content/diff views correctly
- a failed directory load produces a local error state and retry path
- session refresh no longer eagerly requests the full file list

### 12. Constraints

- keep the route name `/api/sessions/:id/file/list`
- do not preserve the old flat recursive response shape
- keep path safety anchored to the session cwd
- keep the UI usable on desktop and mobile
- keep the redesign focused on browsing behavior rather than broader workspace authorization changes

## Out of Scope

- recursive subtree responses or configurable `depth`
- consulting nested `.gitignore` files
- changing file read authorization semantics
- replacing the existing file read or git diff endpoints
- full-text file search changes
- broader workspace UI redesign beyond the file browser tree needed for this behavior

## Acceptance Criteria

The change is successful if:
- `/api/sessions/:id/file/list` returns only one directory level at a time
- the UI can browse the repository by expanding directories lazily
- large repositories no longer require a full recursive file-list payload up front
- root `.gitignore` entries are hidden from the browser tree
- session refresh no longer eagerly loads the entire file list
- file open behavior still works for explicitly selected paths
