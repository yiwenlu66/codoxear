## ADDED Requirements

### Requirement: File path resolution SHALL return a typed outcome with five distinct cases

`_resolve_client_file_path` SHALL return a `FileResolution` value with `status` in `{ok, not_found, dead_symlink, permission_denied, outside_allowed_root}`. Each non-`ok` status SHALL carry a short `detail` string and, where applicable, the original failing path component.

#### Scenario: Valid absolute path resolves to ok

- **WHEN** the user requests an existing readable file by absolute path
- **THEN** `_resolve_client_file_path` returns `FileResolution(status="ok", path=<resolved>, detail="")`

#### Scenario: Dead symlink is reported as dead_symlink

- **WHEN** the requested path is a symlink whose target does not exist
- **THEN** the resolver returns `FileResolution(status="dead_symlink", path=None, detail=<contains target text>)`

#### Scenario: Unreadable parent directory is reported as permission_denied

- **WHEN** a parent directory along the resolved path lacks execute permission for the server process
- **THEN** the resolver returns `FileResolution(status="permission_denied", ...)` rather than `not_found`

#### Scenario: Path escaping the session cwd is reported as outside_allowed_root

- **WHEN** the relative path resolves outside the session's cwd
- **THEN** the resolver returns `FileResolution(status="outside_allowed_root", ...)` and the file is not opened

### Requirement: File-related HTTP endpoints SHALL surface the resolution reason

The endpoints `/api/files/inspect`, `/api/sessions/<id>/file/read`, `/api/files/blob`, and `/api/sessions/<id>/file/blob` SHALL include a `reason` field in their JSON error body for any non-`ok` outcome and SHALL choose status codes by reason: `404` for `not_found` and `dead_symlink`, `403` for `permission_denied`, `400` for `outside_allowed_root`. The current generic `"file not found"` message SHALL not be returned for non-`not_found` cases.

#### Scenario: Inspect of a dead symlink returns reason dead_symlink

- **WHEN** the client posts to `/api/files/inspect` with a path that is a dead symlink
- **THEN** the response is HTTP 404 with `{"error": "...", "reason": "dead_symlink"}`

#### Scenario: Read of a path with permission denied returns 403

- **WHEN** the client GETs `/api/sessions/<id>/file/read?path=<unreadable>`
- **THEN** the response is HTTP 403 with `"reason": "permission_denied"`

### Requirement: The file viewer UI SHALL display the reason returned by the server

The browser file viewer SHALL display the server's `reason` field in a human-readable form when a non-`ok` response is received, and SHALL distinguish at minimum `not_found`, `dead_symlink`, and `permission_denied` so the user can tell why the file is not visible.

#### Scenario: Dead symlink path shows a dedicated message

- **WHEN** the user opens a path that the server reports as `dead_symlink`
- **THEN** the file viewer shows "Symlink target does not exist" (or equivalent) instead of the generic "file not found"
