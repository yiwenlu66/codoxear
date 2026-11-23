# EPISTEMIC — transcript window / paging

## Phenomenon to explain
After the reader moves back into history, (a) scrolling down cannot reach the
session's last message, and (b) the working bubble renders after an old message
rather than at the log's end.

## Live mechanism (supported, reproduced)
The transcript renders a **window** of the log, and two paths install a window
whose last row is not the log's last event:

- paging upward (`prependOlderEvents`) bounds the DOM by deleting rows from the
  **bottom** (`app_transcript.js:852`), which removes the tail once the window
  exceeds `CHAT_DOM_WINDOW + OLDER_PAGE_LIMIT` = 320 rows (5–6 older pages);
- cursor windows (`/messages/window`) are installed by replacement and the
  server's `has_newer` is discarded, so they have no forward edge at all.

After that, four tail-scoped surfaces keep asserting the DOM bottom is the log
end: the typing/subagent bubble is inserted at `bottomSentinel`; "scroll to
bottom" is pure geometry; `jumpToLatest` first *claims* attachment
(`markLiveTail`) and only then reloads; the optimistic send echo bypasses the
attachment gate and its top-trim flips the claim back to true.

The single architectural defect behind all four: **whether the window reaches the
log's end is a content fact stored as mutable placement state
(`renderedAtLiveTail`, six writers, two meanings) instead of being derived from
the installed window.**

## Ruled out (with the evidence that ruled them out)
- *The server cannot page forward.* False — `read_chat_window_around` already
  returns `has_newer`, and `window?cursor&before=0&after=N` yields forward pages.
  What is missing on the server is only the right-edge **live cursor**, needed to
  rebase the transport on attach.
- *The jump button is broken.* False — `jumpToLatest` → `openSession(forceRender)`
  does replace the DOM with the tail. The button is the only working way back;
  the report is about scrolling. The claim-then-reload order still makes a failed
  or aborted reload permanent (jump button hidden, live rows spliced), which is
  the state the reporter describes as sticky.
- *The DOM bound is too small.* Raising `CHAT_DOM_WINDOW` only delays detachment;
  the missing edge is the defect, not the ceiling.
- *The bubble can be placed correctly by picking a better row.* There is no row in
  a detached window that means "the agent is working here"; the tail is absent.

## Claim currently justified (strength)
High: the mechanism, the entry paths, and the ordering/placement outcomes (a real
run of the real modules, table in OPS.md). Medium: the exact user-visible
consequences of the *permanent* stranded state (steps 6–7) — the mechanism is
proven in the harness, but I did not reproduce the live occurrence of a failed or
aborted tail reload in a browser.

## Most valuable open question
Whether the reader's expected "scroll to bottom" is *continuous paging* through
the gap or an instant jump to the newest message. That decision sets whether the
fix adds a forward paging lane (M3) or only an attach-on-edge (M2). It is listed
as an open decision in the plan rather than silently chosen.
