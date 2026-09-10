# EPISTEMIC

## Phenomenon

The transcript exposes only a compact aggregate for active children unless the user activates a hidden-details control. The idle activity bubble has no child records. In parallel Pi workflows, the scanner also collapses multiple running steps into one top-level workflow record.

## Accepted mechanism

The installed Pi producer's lifecycle-v3 `status.json` makes the distinction explicit: top-level state and PID describe the workflow runner, while `steps[i].status`, `agent`, `model`, `toolCount`, and `tokens.total` describe each child. Parallel children transition to `running` and complete independently; a chain leaves future steps `pending` while only its current child is `running`. Modern Pi records must therefore project one row per `running` step. Legacy records whose steps carry no lifecycle status retain the former one-record best-effort projection.

The session-list response already includes `subagents_running` and `subagent_details`, and its ETag hashes the complete JSON. A telemetry-only change therefore produces a fresh response even when the count is unchanged. The selected-session store must own both count and detail records in one atomic patch. The transcript projection subscribes one callback to both fields and becomes the sole writer of busy/idle child lines.

Both activity bubbles retain their compact aggregate first line and automatically render the supplied records beneath it. Lines contain only available facts: role, model, tool count, and cumulative `tokens used`; they never synthesize placeholder rows to make the detail length match the count. Session switching clears count and details together.

Codex and Claude Code retain scanner-provided best-effort records; absent telemetry is omitted rather than labeled unknown.

## Current claim

The correction now preserves one end-to-end authority chain: lifecycle-v3 Pi `running` steps → public `subagent_details`/count → catalog → atomic selected-session `subagentDetails`/`subagentsRunning` → transcript projection → shared busy/idle line renderer. Same-count telemetry changes remain visible because the detail array is a first-class store field. The renderer contains no expansion state or control semantics and never pads missing records with invented unknown rows.

Docker behavior confirms two parallel children remain distinct with their own role/model/tool/cumulative-token telemetry, the summary remains compact, idle/busy transitions retain the lines, and session-switch state removes them. Codex/Claude Code best-effort records continue through the same formatter with absent fields omitted.

## Residual boundary

The deterministic producer fixture matches the installed Pi lifecycle-v3 schema and retained real status artifacts. A future producer changing step lifecycle names or telemetry fields would require a schema update; current browser and scanner behavior is fully covered for the installed contract.
