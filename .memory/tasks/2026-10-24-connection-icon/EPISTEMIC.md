# EPISTEMIC — connection icon

## What is being explained

A phone screenshot showed a two-line bordered panel ("Connection unavailable —
showing cached data. Updates retry automatically.") wedged between the topbar and
the transcript. The user's ask: an offline icon suffices.

## Current model

The banner was not merely verbose; it was structurally mispriced. It was a block
sibling in `main`'s flex column, so its height came from the sentence and its
wrap, and it therefore consumed volume. Nothing bounded it: at 390 px the same
sentence is two lines (56 px measured), and iOS dynamic type or a longer
translation grows it without limit. Connection state is chrome, so it belongs at
chrome's scale.

The three internal reasons are evidence channels, not UI states. `navigator.onLine`
is only trustworthy when it says *no link*; the outage class that actually
happens (VPN tail, captive portal, paused server) leaves `onLine === true` while
nothing reaches the server, so the app has two more channels: a cache-served 200
marked `X-Codoxear-Stale` (proof the payload is not fresh, no threshold) and two
consecutive transport failures (threshold exists so one flaky poll does not
flash). They differ in detection and clearing, so they stay separate in code; the
only user-visible consequence was wording, and wording is now hover + sr-only
text. Collapsing them would either lose the stale admission (cache content read
as fresh) or the no-flap threshold.

## Invariants that must not be traded away

- A cache-served 200 must keep the indicator up. It is evidence about the cache,
  and clearing on it is the original bug this chain exists to prevent.
- One stale answer is proof; transport failure needs two.
- `app_network.js` is the single writer of visibility, `title`, and spoken text;
  the shell owns only the static glyph markup.
- The indicator must never shift `.topActions`. It lives inside `.pill` for that
  reason — measured: `actionsLeft` unchanged at 276 px in both states.
- Visibility, not hue, encodes state (theme invariant); ink color, no filter.

## Ruled out

- Second glyph for "cached" vs "offline": the distinction is an evidence channel,
  and on touch there is no hover to read a legend; send/queue refusal toasts
  already speak the actionable sentence at the moment it matters.
- Click-to-expand sentence: it would turn an indicator into a control with a hit
  area, focus ring, and hint entry, buying information nobody asked for.
- Truncated wifi arcs as the glyph: measured at 18 px, fragments read as noise.
  The full fan + slash is what stays recognizable.

## Second correction: the evidence source

The first iteration fixed the icon's *presentation* and left its *evidence*
where it was: the poll loop. That is the wrong owner. Connection state is a
property of the client, and every HTTP response is evidence about it, so the
producer belongs in the transport layer (`app_api`), not in whichever
session-scoped poll happened to complete. A poll path that reports transport
state couples the indicator to a scheduling cadence: backoff, a suspended tab, or
a boot sequence each delay it for reasons that have nothing to do with the
network.

Consequence for the model: "which requests report" is no longer a per-feature
decision that can be forgotten. A new feature that calls `api` reports
automatically, and a request that is never made reports nothing (so a skipped
refresh cannot clear the indicator, structurally rather than by a rule).

## Verified vs assumed

Verified by measurement: transcript recovered 56 px (chatTop 156→100,
chatHeight 632→688); title absorbs exactly 26 px; topbar and action cluster
unchanged; stale/offline/recovery transitions on a frozen server; boot-to-icon
latency 8045 ms → 3205 ms on a frozen server; 2517 tests pass with one
pre-existing unrelated failure.

Assumed, not measured: iOS dynamic-type scaling of the old banner (why it can
grow without bound) — plausible from the CSS but not exercised.

## Most useful next question

Does the icon read as "not live" to someone glancing at a phone without hovering?
Only user feedback answers that; the glyph choice was made on rendered evidence
at 18 px, not on comprehension testing.
