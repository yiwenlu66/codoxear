# Simple Mode Skill Design

## Goal

Create a lightweight `simple-mode` skill that, when manually invoked, shifts the assistant into a faster and more minimal response mode for the rest of the current conversation until the user explicitly exits that mode.

The skill should reduce the influence of auto-invoked startup or process-heavy skills as much as realistically possible, while preserving higher-priority system, safety, and direct user instructions.

## Non-Goals

- Do not claim to override system-level instructions.
- Do not claim to disable the platform's skill loading mechanism at the infrastructure level.
- Do not introduce new product capabilities beyond response-style and workflow suppression.
- Do not automatically restore normal mode without an explicit user request.

## User Intent

The user wants a "fastest and cleanest" mode for direct question answering. In practice, this means:

- prefer answering immediately
- avoid proactive planning, checklists, or workflow ceremony
- avoid loading or following startup-injected methodology skills unless clearly required
- remain in this mode across subsequent turns
- keep explicit boundaries about what cannot be overridden

## Recommended Approach

Implement a single skill named `simple-mode`.

This skill acts as a persistent behavioral protocol rather than a feature module. Its instructions should tell the assistant to:

1. enter a persistent minimal-response mode when the skill is manually invoked
2. avoid proactively invoking startup, workflow, planning, brainstorming, or process-enforcement skills
3. answer directly, briefly, and with minimal ceremony by default
4. ask follow-up questions only when necessary to avoid material misunderstanding
5. continue using higher-priority system, safety, repository, and direct user instructions
6. remain in this mode until the user explicitly says to exit simple mode

## Alternatives Considered

### Alternative 1: Dual skill design

Use `simple-mode` to enter and `exit-simple-mode` to exit.

Pros:
- clearer entry/exit semantics
- easier discoverability for exit path

Cons:
- more moving parts than the user requested
- unnecessary overhead for an intentionally minimal feature

### Alternative 2: Aggressive override language

Write the skill as if it can fully ignore all injected skills.

Pros:
- stronger rhetorical suppression of workflow-heavy behavior

Cons:
- semantically inaccurate
- likely to conflict with higher-priority instructions
- creates false expectations

The single-skill realistic version is preferred because it is simpler and more truthful.

## Skill Semantics

### Entry

The user manually invokes `/simple-mode`.

### Active behavior

While active, the assistant should:

- default to direct answers first
- minimize planning language, process framing, and meta-commentary
- avoid proactively invoking skills just because they might apply
- avoid generating todo lists, plans, or multi-step structures unless the user asks for them or the task genuinely requires them
- keep responses short and operational
- still investigate or verify when needed, but do so with the lightest viable process

### Boundaries

While active, the assistant must still obey:

- system instructions
- safety constraints
- repository or project instructions
- explicit user requests that call for a different process

### Exit

The user exits by saying a clear phrase such as:

- "exit simple mode"
- "leave simple mode"
- "return to normal mode"

The skill should document these phrases as the normal exit path.

## SKILL.md Structure

The `simple-mode/SKILL.md` file should contain:

1. YAML frontmatter
   - `name: simple-mode`
   - a trigger-focused `description`
2. Overview
   - what simple mode is
   - what problem it solves
3. Activation effect
   - persistent minimal mode semantics
4. Behavior rules
   - direct answers, minimal workflow, no proactive startup skill usage
5. Boundaries
   - cannot override higher-priority rules
6. Exit instructions
   - explicit phrases the user can use
7. Red flags
   - things the assistant should avoid in this mode

## Draft Description Strategy

The description should focus on when to use the skill, not the internal workflow. A suitable direction is:

"Use when the user wants a persistent minimal-response mode with fewer workflow prompts, less process overhead, and more direct answers across the rest of the conversation."

This keeps the description searchable without over-describing the implementation.

## Red Flags To Include

The skill should explicitly warn against these behaviors while active:

- loading methodology skills by default
- expanding into planning mode without necessity
- creating todo lists for straightforward requests
- adding heavy framing before giving an answer
- pretending to override higher-priority system rules

## Testing Strategy

Because this is a skill-definition task, validation should be lightweight and honest:

1. verify the skill text is internally consistent
2. verify the frontmatter follows naming and description rules
3. verify the behavior is framed as best-effort suppression, not absolute override
4. optionally test with a few sample prompts in a later implementation step

## Implementation Notes

The likely target location is the personal skills directory used by the current Pi setup. The design should avoid assuming infrastructure-level support for true runtime skill unloading.

Therefore, the implementation should focus on behavioral instructions that reduce proactive skill usage rather than claiming architectural disablement.
