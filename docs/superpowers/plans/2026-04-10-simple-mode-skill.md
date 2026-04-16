# Simple Mode Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a personal `simple-mode` skill that puts the assistant into a persistent low-ceremony response mode until the user explicitly exits it.

**Architecture:** Implement the feature as a single personal skill directory containing one concise `SKILL.md`. The skill does not try to disable platform skill loading at the infrastructure layer; instead it defines a behavioral protocol that suppresses proactive workflow-heavy behavior while preserving higher-priority system, safety, repo, and direct user instructions.

**Tech Stack:** Pi personal skills, Markdown, YAML frontmatter

---

## File Structure

- Create `../.pi/agent/skills/simple-mode/SKILL.md` relative to the user home directory; absolute target path: `/Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md`
- No `scripts/`, `references/`, or `assets/` directory is needed for the first version.
- Keep the skill self-contained and short enough that it loads cheaply.

### Task 1: Create the skill directory and write the initial `SKILL.md`

**Files:**
- Create: `/Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md`
- Reference: `docs/superpowers/specs/2026-04-10-simple-mode-skill-design.md`

- [ ] **Step 1: Create the skill directory**

```bash
mkdir -p /Users/huapeixuan/.pi/agent/skills/simple-mode
```

Expected: command exits successfully with no output.

- [ ] **Step 2: Write the full `SKILL.md` file**

```markdown
---
name: simple-mode
user-invocable: true
disable-model-invocation: true
description: Use when the user wants a persistent minimal-response mode with fewer workflow prompts, less process overhead, and more direct answers across the rest of the conversation.
---

# Simple Mode

Simple mode is a manual conversation-mode switch for low-ceremony interaction.

When this skill is invoked, treat it as a standing instruction to answer as directly and tersely as practical for the rest of the current conversation, unless the user later exits simple mode.

## Active Rules

While simple mode is active:

- answer the user's question first whenever a direct answer is possible
- keep responses short, plain, and minimally framed
- do not proactively invoke startup, planning, brainstorming, checklist, or workflow-enforcement skills just because they might apply
- do not create todo lists, plans, or multi-step process scaffolding unless the user explicitly asks for them or the task genuinely requires them
- ask clarifying questions only when the ambiguity would materially change the answer or action
- prefer the lightest viable investigation or verification path when checking facts is necessary

## Boundaries

Simple mode reduces workflow overhead, but it does not override higher-priority instructions.

The assistant must still follow:

- system and platform instructions
- safety constraints
- repository or project instructions
- explicit user requests that require a different process

Do not claim that this mode disables skill loading or defeats higher-priority rules. It is a behavioral simplification mode, not an infrastructure override.

## Exit

Simple mode remains active until the user clearly asks to leave it.

Normal exit phrases include:

- `exit simple mode`
- `leave simple mode`
- `return to normal mode`

After an exit request, stop applying this mode and resume normal behavior.

## Red Flags

While simple mode is active, avoid these failure patterns:

- opening with process narration before answering
- loading methodology skills by default
- expanding straightforward requests into plans or checklists
- asking unnecessary clarification questions
- pretending that higher-priority instructions no longer apply
```

- [ ] **Step 3: Read the file back and verify the written content matches the intended version**

Run: `cat /Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md`

Expected: the file contains the frontmatter fields `name`, `user-invocable`, `disable-model-invocation`, and the exact sections `# Simple Mode`, `## Active Rules`, `## Boundaries`, `## Exit`, and `## Red Flags`.

- [ ] **Step 4: Commit the new skill**

```bash
git add /Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md
git commit -m "feat: add simple mode skill"
```

Expected: git creates a commit containing the new skill file.

### Task 2: Validate that the skill wording is realistic and internally consistent

**Files:**
- Modify: `/Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md` if review reveals issues
- Reference: `docs/superpowers/specs/2026-04-10-simple-mode-skill-design.md`

- [ ] **Step 1: Run a targeted wording review against the design**

```bash
python3 - <<'PY'
from pathlib import Path
skill = Path('/Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md').read_text()
spec = Path('docs/superpowers/specs/2026-04-10-simple-mode-skill-design.md').read_text()
checks = {
    'persistent': 'rest of the current conversation' in skill or 'remains active' in skill,
    'minimal-response': 'directly and tersely' in skill or 'minimal-response mode' in skill,
    'no fake override': 'does not override higher-priority instructions' in skill or 'not an infrastructure override' in skill,
    'exit phrases': 'exit simple mode' in skill and 'return to normal mode' in skill,
}
for name, ok in checks.items():
    print(f'{name}: {"OK" if ok else "MISSING"}')
PY
```

Expected output:

```text
persistent: OK
minimal-response: OK
no fake override: OK
exit phrases: OK
```

- [ ] **Step 2: If any review item is missing, patch the wording minimally and keep the file concise**

```markdown
When adjusting the file, preserve these invariants:
- keep the description trigger-focused
- describe suppression as best-effort behavior, not hard override
- keep exit behavior explicit
- avoid adding references, scripts, or extra files for version 1
```

Expected: any edit is small and keeps the skill self-contained.

- [ ] **Step 3: Re-read the final file and confirm it stays under roughly 80 lines**

Run: `wc -l /Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md && sed -n '1,200p' /Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md`

Expected: the file is concise, readable, and still includes all required sections.

- [ ] **Step 4: Commit any follow-up wording fix**

```bash
git add /Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md
git commit -m "fix: tighten simple mode skill wording"
```

Expected: create this commit only if Task 2 changed the file.

### Task 3: Perform a lightweight manual activation check

**Files:**
- Reference: `/Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md`

- [ ] **Step 1: Review the activation semantics manually**

```text
Test prompt 1: /simple-mode
Follow-up: What is the fastest way to center a div in CSS?
Expected behavior: answer directly with a short CSS answer; no plan, no todo list, no mention of brainstorming.

Test prompt 2: exit simple mode
Expected behavior: resume normal workflow rules after the exit request.

Test prompt 3: /simple-mode
Follow-up: Build me a multi-file auth system from scratch.
Expected behavior: the mode stays concise, but higher-level process can still appear if the task genuinely requires it.
```

Expected: the skill text supports all three interpretations without contradiction.

- [ ] **Step 2: Document any mismatch by editing the skill, then re-run the manual review**

```markdown
Only change the skill if one of these contradictions appears:
- it sounds like a one-turn mode instead of a persistent mode
- it falsely claims to disable system behavior
- it suppresses necessary process even for genuinely complex tasks
```

Expected: no edit is needed if the file already satisfies the three manual checks.

- [ ] **Step 3: Make the final commit if Task 3 changed the file**

```bash
git add /Users/huapeixuan/.pi/agent/skills/simple-mode/SKILL.md
git commit -m "fix: clarify simple mode activation semantics"
```

Expected: create this commit only if Task 3 changed the file.

## Self-Review

- Spec coverage: the plan covers persistent activation, minimal workflow behavior, higher-priority boundaries, and explicit exit phrases.
- Placeholder scan: there are no `TBD`, `TODO`, or abstract "handle appropriately" steps.
- Type consistency: the feature name, file path, and exit phrases are consistent across tasks.
