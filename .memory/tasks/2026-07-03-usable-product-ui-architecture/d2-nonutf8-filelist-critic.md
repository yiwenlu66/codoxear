# Audit Verdict: **Commit**

The audit's core hypothesis — that committing creates a *worse* product contract — is **false**. The fix removes a hard failure and aligns walk-mode behavior with the semantics already shipped elsewhere. Non-openability of non-UTF-8 results is a pre-existing limitation of the non-git browser, not a regression this change introduces.

## What the fix does (mechanism verified)

Pre-fix, a single non-UTF-8 filename (byte `0xff` → lone surrogate `\udcff` from `os.walk`) crashed the **entire** `file/list` and walk-mode `file/search` response at `json...encode("utf-8")`. Reproduced directly:

- Pre-fix: `UnicodeEncodeError: 'utf-8' codec can't encode character '\udcff'` → HTTP 500
- Post-fix: `path_json_text("bad\udcffname.bin")` → `'bad\\xffname.bin'`, JSON/UTF-8 safe

## Why it is not a worse contract

**1. The baseline being replaced is a total 500, not a working-but-openable state.** With any non-UTF-8 file present, the whole endpoint failed — *every* file in that directory (including all normal UTF-8 files) was inaccessible. Post-fix: 200, all UTF-8 files openable, non-UTF-8 files visible. Strictly better.

**2. Non-openability is not introduced here.** The non-git file browser has never had a reversible token channel:
- `_finish_file_search` emits only `{"path", "score"}` — **no `api_path` in any mode** (`file_search.py`), so the client never receives a token from search/list.
- The non-git read/blob/download endpoints accept only `path`; `_git_or_plain_query_path` reads `path_token` **only when `git_path=1`** (`file_get_routes.py:207-215`). The client mirrors this — it sends `path_token` only when `gitPath` is true (`app_file_viewer.js:1050-1052`).
- Confirmed open failure: `resolve_session_path(base, "bad\\xffname.bin")` builds `(base / Path("bad\\xffname.bin")).resolve()`; `require_existing_file` `stat()`s it → `FileNotFoundError` because the literal `\xff` string never matches the on-disk `0xff` byte (`path_runtime.py:43-70`).

**3. It matches existing git-mode search.** `search_git_relative_files` already returns lossy, non-openable display for non-UTF-8 names via `errors="replace"` (byte `0xff` → U+FFFD `�`). The walk fix makes walk mode consistent with that, using a *more informative* codec (`\xff` vs. `�`).

**4. Forward compatible — no lock-in.** `path` keeps display semantics (identical to the git diff layer's `path` field). A future openability fix adds `api_path`/`path_token` to entries; that is purely additive and does not require changing the `path` field.

**5. Minimal and safe.** Only walk + list changed; git-mode search untouched. `git_ops` is a leaf module — no import cycle (verified). 8/8 `test_file_list.py` pass; 117 related tests pass.

## Tests: adequate

Both new tests exercise the real mechanism (create a `0xff`-byte filename, assert no `\udcff`/`\udc80` surrogates survive, assert the JSON-escaped `bad\\xffname.bin` byte sequence is present in an actual `json.dumps(...).encode("utf-8")`). They validate exactly the failure mode (response-body encoding) and match the git-layer `backslashreplace` convention. Linux-first surrogateescape assumptions and the "/tmp is non-git ⇒ walk mode" convention are consistent with the pre-existing tests in the same file.

## Residual risks (non-blocking, out of D2 scope — do **not** gate this commit)

- **R1 — cwd path itself non-UTF-8 still 500s.** Both `_handle_session_file_list` and `_handle_session_file_search` emit `"cwd": str(base)` unguarded, whereas `git_routes.py` wraps it via `path_json_text(cwd)`. Pre-existing inconsistency, not introduced here; D2 is scoped to a filename *inside* cwd. Recommend a separate follow-up to wrap those two `cwd` fields.
- **R2 — true openability of non-UTF-8 results** requires adding `api_path` to search/list entries plus `path_token` acceptance on the non-git read/blob/download (and write) endpoints, plus client wiring. This is a cross-cutting feature (response schema + endpoints + `app.js`), correctly excluded from a "no 500" blocker. Recommend as a tracked enhancement.

No revise required: there is no *minimal* variant that also restores openability — a non-UTF-8 name cannot be represented losslessly in a JSON string field without a side-channel token, which is the R2 feature.