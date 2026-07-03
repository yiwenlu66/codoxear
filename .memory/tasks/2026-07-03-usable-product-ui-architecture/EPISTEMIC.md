# Epistemic model

## Phenomenon
Codoxear passes its suites, but "fully usable product with decent UI" needs user-perspective evidence. Walkthrough round 1 (desktop 1440x900 + mobile 390x844, sandbox 19083, no real backend) is complete; real-backend round pending.

## UX defect ledger (round 1: no-backend flows)

Fixed this round (verified in browser after fix):
- [FIXED fc620e6] Failed-launch transcript rendered raw ANSI escapes as garbage (`←[35m...`) — most important error text unreadable. Now stripped at `launch_failure_tail`. Evidence: d08 (before), d09 (after).
- [FIXED cefd7d6] Unattended button on failed launch fired real config route -> toast `unattended load error: unknown session`. Now disabled with label "Failed launch has no unattended mode" (same family as earlier Details/file-viewer fixes).

Open, ranked:
- IMPAIRING: Empty app state gives zero guidance — blank sidebar, no "create your first session" CTA in main area (d02). First-run experience is a dead end until user finds the + icon.
- POLISH: Composer send button is rendered fully saturated/active while disabled with no session selected (d02); disabled state not visually communicated.
- POLISH: Backend tabs in New Session are icon-only with a subtle active ring (d03-d05); active-tab contrast is weak, no text label anywhere in the modal saying which backend is selected.
- POLISH: Toast placement/styling is a bare gray text line at the top-left of the chat area (d10); easy to miss, looks unstyled.
- POLISH: Provider/model dropdown truncates on 390px ("openai-api/defa...") (m03). Acceptable but could use tighter label.
- NOTE (not a defect): mobile New-session button lives only in the sidebar drawer; hamburger-first flow is standard.
- NOTE: session-name placeholder auto-fills from cwd basename ("workspace"); good.

Verified good this round:
- Login, sidebar grouping (NEEDS REVIEW + failed badges), recovery panel actions (New like this carries preset; Dismiss works via API; Copy details), Help content accuracy, Settings dialog, mobile layout/wrapping/swipe-delete affordances, Pi/Claude tab adaptation (Fast hidden for Pi), recent-cwd prefill in New Session.

## Architecture debt status
- [DONE 889a8e7] tests/test_message_route_source.py -> test_message_routes.py with injected deps (worker contract 1); no monkeypatch left.
- IN FLIGHT: worker contract 2 (bdfe768c): tests/test_transcript_export.py conversion.
- Remaining app.js concentrations unchanged: chat search/navigation orchestration, new-session dialog, queue/recovery panels.

## Open questions (highest value first)
1. Real Pi session end-to-end in sandbox (send/response/queue/interrupt/search/file-viewer) — needs pi CLI installed in container + Pi config copied per docker-test skill. Nothing about live-session UX is measured yet: transcript rendering with real content, busy indicators, scroll behavior, queue drain are all unobserved on this HEAD.
2. Does the empty-state fix warrant a first-run CTA panel? (Design choice; implement small.)
3. Which app.js concentration next after test conversions: new-session dialog is the largest self-contained block.

## Ruled out
- "Suites green => usable": falsified again this round by the ANSI and unattended defects, both invisible to pytest.
