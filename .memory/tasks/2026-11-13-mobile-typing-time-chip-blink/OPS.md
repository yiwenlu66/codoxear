# OPS
- 2026-11-13: Read chip writer + predicate; confirmed one writer, no timer.
- 2026-11-13: Docker harness /tmp/tip-probe/{build.sh,probe*.js}: working-tree build
  (esbuild bundle + docker/sandbox.Dockerfile), container `tip-probe-19717` on
  127.0.0.1:19717 (never 8743), session from `tests/fixtures/compaction_broker.py`
  (24 rendered rows), Playwright chromium via agent-browser's playwright-core with
  iPhone metrics + `Emulation.setSafeAreaInsetsOverride`.
- probe1: chip none at tail; inline-flex after -600 wheel; none again at bottom; typing
  at bottom produced zero chip mutations.
- probe2: 8px-step sweep from the tail; chip appears at ~82px from the bottom;
  no row lacks `dataset.ts`.
- probe3: --40px nudge (detached, still <=80px) then focus -> `handleComposerFocus`
  re-attaches (autoScroll=true) and the chip never appears through keyboard shrink+typing.
- probe4: `--appH` geometry change alone produced no chip mutation and no scroll event.
- probe5: fake `visualViewport` (height 500, offsetTop jitter, height churn 500/564)
  driving the real `onViewportShift` path: chip stayed hidden through typing; the guard
  kept the chat at the bottom (top == scrollHeight - clientHeight every step).
- screenshot /tmp/tip-probe/top-strip.png: the element is the grey date/time band at
  y=108, directly under the 103px topbar.
- 2026-11-13: Implemented intent attribution in createTranscriptScrollRuntime
  (+ view-controller `noteUserScroll`, `pointerdown` binding on `#chat`,
  shortcut tagging). Tests: new tests/test_transcript_scroll_intent.py (7
  behavioral cases through the real runtime in Node VM); updated
  test_chat_transcript_runtime.py for the new contract (unprompted upward event
  does not detach; scrollToBottom targets the exact bottom offset).
- 2026-11-13: tests/test_sse_battle_advanced.py passed whole module sources as
  one `node -e` argument; the combined source was ~124.5KB against the 128KiB
  MAX_ARG_STRLEN and my +2KB tipped it to "Argument list too long". Switched its
  three harness runs to a temp-file runner.
- 2026-11-13: Full suite: 1 failed, 2829 passed — the failure is
  tests/test_hint_mode_coverage.py "draftUndoBtn", pre-existing (reproduced on a
  clean `git archive HEAD` copy in /tmp/headcheck).
- 2026-11-13: Behavioral Docker A/B on ports 19722 (HEAD) / 19721 (working
  tree), same probe /tmp/tip-probe/intent.js. Results in EPISTEMIC.md.
