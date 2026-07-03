# Epistemic model

## Phenomenon
Codoxear passes its suites, but "fully usable product with decent UI" needs user-perspective evidence. Walkthrough round 1 (desktop 1440x900 + mobile 390x844, sandbox 19083, no real backend) is complete; real-backend round pending.

## UX defect ledger (round 1: no-backend flows)

Fixed this round (verified in browser after fix):
- [FIXED fc620e6] Failed-launch transcript rendered raw ANSI escapes as garbage (`←[35m...`) — most important error text unreadable. Now stripped at `launch_failure_tail`. Evidence: d08 (before), d09 (after).
- [FIXED cefd7d6] Unattended button on failed launch fired real config route -> toast `unattended load error: unknown session`. Now disabled with label "Failed launch has no unattended mode" (same family as earlier Details/file-viewer fixes).

Fixed in live round 2 (real Pi session):
- [FIXED af1fea4] Live message polling 500 on every bound-cursor poll (dropped max_bytes default in route extraction). Live streaming verified clean post-fix (d18).
- [FIXED acc232c] rollout_idle NameError family: eight pi/cc helper imports dropped in decomposition; queue path surfaced it live (`queue error: name 'pi_assistant_thinking_count' is not defined`). Queue -> drain verified post-fix (d21).

Verified good in live round 2: session create/auto-select, send/response, second send via live cursor, queue badge/toast/auto-drain, interrupt (Esc) ends turn, loaded+all search counts, file viewer with honest Monaco-timeout fallback, live token chip (d13-d25).

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
- [DONE 0a42e01] tests/test_transcript_export.py conversion (worker contract 2).
- [DONE 0697a3e] tests/test_sessions_pending_log_idle.py conversion (worker contract 3); ~46 internal monkeypatch seams total removed.
- Remaining app.js concentrations unchanged: chat search/navigation orchestration, new-session dialog, queue/recovery panels.

## Open questions (highest value first)
1. Mobile round on the LIVE session (busy indicators, scroll-follow, composer ergonomics at 390px) — desktop live round done.
2. Empty-state first-run CTA + composer disabled affordance + backend-tab active contrast + toast styling (ledger polish items).
3. Next app.js concentration: new-session dialog extraction; next monkeypatch cluster: tests/test_file_inspect.py (largest remaining).
4. Monaco-from-CDN timeout in sandbox produced fallback: decide whether to vendor monaco or accept documented fallback.

## Ruled out
- "Suites green => usable": falsified again this round by the ANSI and unattended defects, both invisible to pytest.
