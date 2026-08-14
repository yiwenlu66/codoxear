# Operational Lessons

Hard-won lessons from agent process failures. Each entry is a rule derived
from a specific incident where the agent was wrong.

## Verify the interface before diagnosing the implementation

Incident: Agent tested `/api/sessions/<id>/messages` (doesn't exist), got 404,
concluded "server broken, route not registered." Real endpoint is
`/messages/tail`. Wasted 20 minutes investigating a non-bug.

Rule: Before concluding anything is broken, verify the exact URL, API key,
parameter names, and response schema against the source. Read the route
registration, not just the handler.

## Follow the user's explicit design direction exactly

Incident: User said "blocked is orthogonal to busy/idle, show actual
busy/idle." Agent added a `blocked→suppressed` override anyway, directly
contradicting the instruction. Had to revert after user called it out.

Rule: When the user states a design rule, implement exactly that. Do not
"improve" it with additional logic the user didn't ask for.

## Investigate the user's specific case, not the general case

Incident: User reported "new sessions don't appear in tmux." Agent ran one
test, saw it work, and dismissed the report. The user's specific session
(dexgem_blogs) was actually in tmux — but the correct response was to
investigate THEIR observation, not to prove the general case works.

Rule: "It works for me" is not a valid response to a bug report. Reproduce
the user's exact scenario. If you can't reproduce it, say so and ask what
they see.

## Check the response schema before parsing

Incident: Agent checked `d["messages"]` when the API returns `d["events"]`.
Concluded "0 messages, parser broken." The parser was fine; the agent read
the wrong key.

Rule: When parsing an API response, check the actual key names in the
response before extracting data. Print the full response on first contact.

## Compound errors cascade

The pattern across all incidents: agent forms hypothesis → tests wrong thing →
gets wrong result → builds wrong conclusion → presents with confidence →
next hypothesis builds on the wrong conclusion. Each error compounds.

Rule: When a result contradicts expectation, stop and verify the test itself
before building on the result. The test is more likely wrong than the system.

## Docker daemon iptables failure

Incident: `docker_verify.sh` fails with "iptables: No chain/target/match."
Fix: `sudo systemctl restart docker`, wait 5 seconds, retry.

Rule: Add this to the deploy runbook as a known transient failure.

## Deploy worktree dirty state

Incident: Deploy script rebuilds the bundle inside the deploy worktree,
making it dirty. Next deploy refuses to update.

Rule: Always clean before deploy:
`git -C ~/.local/share/codoxear/deploy checkout -- . && git -C ~/.local/share/codoxear/deploy clean -fd`
