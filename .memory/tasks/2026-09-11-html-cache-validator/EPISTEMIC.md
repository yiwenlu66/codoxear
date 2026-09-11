# EPISTEMIC

## Phenomenon

After a normal release that changed only frontend assets (not `index.html` bytes), a user's normal reload kept rendering the previous release's UI (displaced subagent activity label). The revalidation returned a "not modified" answer for HTML whose served representation had actually changed.

## Accepted mechanism

The HTML representation is produced by server-side substitution (`read_static_bytes`: content-hashed asset version + attachment limit into placeholders), but the weak ETag was derived from the raw file's stat. On the real release path the serving root is the deploy git worktree (`python -m` + `WorkingDirectory` put CWD first on sys.path; the pipx venv is never the import root), and git rewrites only content-changed files, so an asset-only release leaves index.html's stat — and therefore the stat ETag — unchanged while the served bytes change. `Cache-Control: no-cache` forces per-load revalidation, the stale ETag yields 304, and old HTML pins old `?v=` immutable asset URLs (max-age=31536000, immutable) so subresources are never re-requested either. Chrome composes the cached body and reports the reload navigation as status 200 with header-only `transferSize` — the discriminating browser evidence, since `responseStatus` alone does not expose the 304.

Second trigger, same mechanism: an attachment-limit-only change (env + restart) alters the served HTML with no file change at all.

## Current claim (validated)

The fix makes the HTML validator representation-derived: `W/"<sha256(final substituted bytes)>"`, computed from the single `read_static_bytes` call shared by the 304 decision and the 200 body. Non-HTML assets keep the stat fast path (304 without reading) and the immutable policy; the hash is taken before content coding, so one weak validator is shared across gzip/identity, matching prior cross-encoding behavior.

Docker evidence:
- Real-HTTP repro matrix: pre-fix 304-for-changed-representation on both triggers and both HTML routes; post-fix 200 on changed representations, 304 on unchanged (immediately and after restart, gzip included), non-HTML validators and immutable policy untouched.
- Browser upgrade (a206ca5d → 5d62d020±fix, one persistent profile, index.html stat preserved, plain `location.reload()`): buggy keeps old HTML/assets/layout (gap 90px); fixed re-downloads HTML (transfer 3682), fetches both new immutable assets, renders `▸2 subagents working` beside its marker at 8px.
- Browsers holding old stat-format ETags take a one-time 200 after the fix deploys (no cached client can be pinned by the format change: mismatch ⇒ full response).

## Residual boundary

- Live-path triggers were established from disk metadata + git diffs only (no live HTTP probes by task rule); Docker reproduction is the behavioral stand-in and uses identical serving semantics (workdir-imported package, unchanged-file stat preservation).
- Latent, out of scope: the built wheel omits `static/dist/` (package-data lists no `static/dist/*`), so the bundle exists only in the deploy worktree serving root; if the unit ever lost `WorkingDirectory` (CWD import) the live app would 404 its entry module. Reported to the parent as a deployment-scope risk, not fixed here.
- Within one server process, `static_asset_version` stays memoized (pre-existing behavior): editing assets on a live server without restart does not refresh the version; unchanged by this fix.
