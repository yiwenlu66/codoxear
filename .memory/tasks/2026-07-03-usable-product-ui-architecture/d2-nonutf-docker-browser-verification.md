# D2 non-UTF file list/search Docker browser verification

Functional commit: `656b7c7 Serialize non-UTF file paths in workbench lists`.
Evidence commit to be created after this report.

## Boundary

- Docker browser sandbox: port `19133`, name `codoxear-d2-browser-19133`, root `/tmp/codoxear-d2-browser-19133`.
- Docker unit sandbox: port `19134`, name `codoxear-d2-test-19134`, root `/tmp/codoxear-d2-test-19134`.
- Host live app dir and live backend/session/tmux state were not used.
- Fixture used one container-only fake control socket session `d2-nonutf`, with cwd `/home/tester/work/nonutf` and raw-byte filenames `bad\xffname.bin` and `src/needle\xfffile.txt` created inside the container.
- Cleanup: browser sandbox stopped through `scripts/codoxear-docker-sandbox stop`; named browser session closed. The fake socket process was restarted once by exact container PID to correct a fixture-only token shape.

## API observations

After discovery throttle elapsed, `/api/sessions` listed `d2-nonutf` with `busy=false`, `log_path=null`, cwd `/home/tester/work/nonutf`, and app dir remained `/home/tester/.local/share/codoxear`.

`GET /api/sessions/d2-nonutf/file/list` returned HTTP 200:

```json
{"ok": true, "cwd": "/home/tester/work/nonutf", "files": ["bad\\xffname.bin", "normal.txt", "notes.md", "src/needle\\xfffile.txt"]}
```

`GET /api/sessions/d2-nonutf/file/search?q=needle&limit=20` returned HTTP 200 with walk mode and `src/needle\xfffile.txt`.

`GET /api/sessions/d2-nonutf/file/search?q=bad&limit=20` returned HTTP 200 with walk mode and `bad\xffname.bin`.

A parser check loaded each JSON body and re-encoded it with `json.dumps(..., ensure_ascii=False).encode("utf-8")`; all succeeded. The bodies contained no `\udc*` surrogate escape and no raw codec error text.

## Browser observation

In a real browser against the Docker server:

- Login succeeded with the sandbox password.
- Session `d2-nonutf` selected.
- File Workbench opened through the `View file` toolbar button.
- Searching `bad` in the file picker rendered option `bad\xffname.bin`.
- The browser DOM check found no `UnicodeEncodeError`, `surrogates not allowed`, or codec-error text, and no bootstrap load error.

Screenshot: `browser-artifacts/d2-filepicker-bad-nonutf.png`.

## Validation

- Targeted local: `128 passed, 56 subtests` for `tests/test_file_list.py tests/test_file_routes.py tests/test_file_inspect.py tests/test_file_picker_search_source.py tests/test_git_ops.py`.
- Full local: `1629 passed, 132 subtests passed in 23.18s`.
- Docker unit: `1628 passed, 1 skipped, 132 subtests passed in 43.69s`.
- Docker smoke/browser sandbox: pre-login `/api/me` 401, post-login `/api/sessions` 200, app dir `/home/tester/.local/share/codoxear`.

## Residual boundary

The verified claim is no-crash/no-raw-codec-error for list/search display. Reversible open/edit/download of non-UTF walk/list results remains a separate additive token-channel feature.
