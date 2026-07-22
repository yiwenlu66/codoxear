# Epistemic model — product experience feedback

## Phenomenon
The refactored product is deployed as an isolated Docker preview and the user is experiencing it directly. This task converts ongoing observations into fixes or queued issues.

## Current mechanism
The preview is a faithful projection of the recovery product against a real workspace. Each user comment is either a localized defect (fix-now) or a nontrivial/ordering-sensitive issue (queued in PROMPT.md §Issues).

## Supported claims
- Preview is reachable, authenticated (distinct cookie from main), and has produced a real protocol-v2 Pi round trip.
- Isolation holds at runtime and (after the cookie fix) at browser-auth boundaries.

## Live uncertainties
- None yet; awaiting first user comment.

## Current claim
Tracker is initialized and ready. No product issue is currently justified.

## Highest-value next question
The first user observation determines whether the product's real-world projection matches its certified behavior.
