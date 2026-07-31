PASS. Blockers: none.

- `tests/test_file_upload.py` no longer imports `codoxear.server`, `unittest.mock.patch`, or drives `server.Handler.do_POST`; the only seam references are explanatory docstring text at lines 5-10.
- Old upload-helper coverage is preserved directly against `codoxear.file_upload`: filename normalization, binary staging, oversize rejection, Unicode filenames/paths, 0600 chmod, index validation, and UTF-8 PTY encoding are covered in `tests/test_file_upload.py:28-186`.
- Route coverage moved to the injected dependency seam: `tests/test_control_routes.py:235-278` builds `ControlRouteDeps`, with success/serialization/bracketed paste at `281-296` and status cases through `441`.
- Scope is test-only: commit diff contains only `tests/test_control_routes.py` and `tests/test_file_upload.py`.
- Non-blocking residual: the status matrix is meaningful but not literally exhaustive; `codoxear/control_routes.py:287-288` handles `session_not_ready_error` raised during `inject_attachment_keys`, and I did not find a direct new test for that post-readiness race branch. Existing tests cover the same 409 status through readiness failures.