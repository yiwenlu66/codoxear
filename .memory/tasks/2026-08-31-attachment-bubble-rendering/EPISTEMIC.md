# Epistemic model — attachment bubble rendering

## Phenomenon
Web-sent messages with attachments show "Attachment N: /abs/path" lines in the
bubble; paths were not clickable, copy omitted the attachment lines, and images
did not render inline.

## Established mechanism
- Server injects attachment prefix into committed text (session_send.py:85-89,
  file_upload.py:159 attachment_inject_text). The Pi log's user event text
  contains the full string; projection serves it verbatim as ev.text
  (verified against the real session log, OPS 2026-08-31).
- Web sends render an optimistic echo from composer text (no prefix). On commit,
  consumePendingUserIfMatches (app_transcript.js) patched only innerHTML + ts,
  leaving (a) copy handler closed over echo text, (b) candidate file-ref spans
  un-upgraded. Both symptoms vanished on reload because fresh rows go through
  makeRow. Single shared root cause for issues 1+2.
- Server path resolution (resolve_session_path / resolve_client_file_path)
  imposes no cwd confinement on absolute paths; inspect/blob endpoints serve
  the uploads dir fine. Server was never the problem.

## Fixes (committed)
- c0747b22: row owns copyText (read at click time, rewritten on commit);
  commit re-runs upgradeCandidateFileRefs on swapped content; dep wired through
  createPendingUserOptions.
- ff40bf64: attachmentDisplayMarkdown in app_markdown.js rewrites the leading
  "Attachment N: /path" block for chat display only (chatMarkdownHtmlCached);
  image ext -> ![basename](path); whitespace/paren paths and non-images stay
  plain (linkified by candidate-ref upgrade). Committed text untouched.

## Ruled out
- Changing the injected text to markdown image syntax at send time: pollutes
  agent prompt/terminal echo, breaks sidebar redaction grammar, filename
  escaping (spaces/parens) corrupts paths, and existing logs keep the old
  format forever so the display projection is needed anyway.

## Design pass outcome (evidence-based)
- Full-size screenshot audit: composer borderless utility buttons and the
  backend-tab 2px active outline are BOTH deliberate, pinned decisions
  (commit 87d4c108; test_issue_archive_ui_contract.py explicitly rejects
  filled tabs and outline-plus-underline). Left untouched — changing them is
  the thrash the user forbade.
- Fixed: queue glyph (hamburger-ambiguous) -> stacked layers (b95b4c99);
  cc.svg viewBox cropped for optical weight parity (d1a60aed).
- Minor pre-existing gap: send with empty text + attachment yields no copy
  button on the settled bubble (makeRow only creates it for non-empty echo
  text). Not addressed.

## Verification status
- Attachment fixes: Docker-verified PASS at 947146f4 (criteria a-d).
- Design changes: final visual verification at 3ddc8a1f (run 13c45322).
- Harness lesson: docker_verify.sh uses the committed bundle as-is; rebuild
  + commit dist/app.bundle.js before any frontend verification (recorded in
  .memory/project/VALIDATION.md).
