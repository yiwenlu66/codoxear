## Objective
Make rendered code-block copy buttons meet the 44x44 CSS-pixel touch-target contract on coarse-pointer companion devices in the 521–880px viewport range.

Done when the CSS/source implementation is committed, local validation passes, Docker/browser proof shows coarse-pointer 521–880px code-copy controls are at least 44x44 without regressions on phone or desktop, clean-room review accepts the slice, and task/project memory records the accepted invariant.

## Workbench
1. Prove the current coarse-pointer 521–880px gap with a failing discriminator before fixing it.
2. Add the smallest CSS rule that gives `.code-copy-btn` a 44x44 target on coarse-pointer devices and enough `<pre>` right padding for that button.
3. Preserve existing phone 44x44 behavior, desktop fine-pointer compact sizing, message-copy behavior, block-local copy semantics, and no horizontal overflow.
4. Validate locally with focused source/CSS tests plus full pytest.
5. Prove in Docker/browser at touch tablet, touch phone, and fine-pointer desktop viewports; run clean-room review and record memory.

## Context
Active checkout: `/home/yiwen/codex-web-product-recovery` on branch `recovery/product-gaps`.
Protected checkout: `/home/yiwen/codex-web` on `main`; do not touch.
Fresh next-slice scout: `/tmp/codoxear-next-slice-after-copy-count.md` and subagent run `37dd1569-6a12-420c-ab26-f5fb67fa8ea1`.
Relevant accepted prior task: `.memory/tasks/2026-07-07-code-block-copy-buttons/`.
Relevant source/tests: `codoxear/static/app.css`, `codoxear/static/app_markdown.js`, `tests/test_code_block_copy_source.py`, `tests/test_mobile_shell_touch_targets_source.py`.
Docker skill: `.codex/skills/codoxear-docker-test/SKILL.md`.

## Task specifications
Current mechanism: `.code-copy-btn` has base `width`/`height`/`min-width`/`min-height` of 30px. The phone media query at `max-width: 520px` overrides it to 44x44 and increases `.md pre` right padding. The coarse-pointer media query sizes generic `button` min-height to 40px and `.icon-btn` to 40x40, but `.code-copy-btn` is a standalone classed button created by `app_markdown.js`; the class-level 30px rule defeats the element-level coarse-pointer rule, and `.code-copy-btn` is not `.icon-btn`. Therefore a touch tablet or small touch laptop at 521–880px can render code-copy controls at 30x30, below the code-copy 44x44 contract recorded in project architecture.

Target mechanism: in a coarse-pointer context, `.code-copy-btn` is at least 44x44 CSS px and the surrounding code block right padding accommodates it. Phone `<=520px` remains at least 44x44. Fine-pointer desktop remains compact (30x30) to avoid desktop UI bloat. The copied payload and code-copy event behavior must not change.

Discriminator should prove the CSS currently lacks a coarse-pointer `.code-copy-btn` 44px rule and/or measured browser size is below 44px at a 521–880px coarse-pointer viewport. A source/CSS discriminator is acceptable for implementation, but final acceptance requires real Docker/browser measurement.

Browser proof must use a deterministic session with at least one rendered markdown code block and the real `.code-copy-btn`. Required viewport evidence:
- coarse-pointer tablet around 768x1024: code-copy button rect width/height >=44, no page-level horizontal overflow.
- touch phone around 390x844: remains >=44, no overflow.
- fine-pointer desktop around 1280x800: remains compact (about 30x30) and no overflow.
Also verify block-local copy still works for at least one code block and message-copy behavior is not changed if practical.

## Constraints
Do not edit/promote/merge protected `/home/yiwen/codex-web` or `main`.
Do not touch live runtime dirs: `~/.local/share/codoxear`, `~/.claude`, `~/.codex`, host Pi logs/sockets, systemd/tailscale.
Docker-only for broker/server/session/tmux/browser verification; avoid port `8743`.
Cleanup must be exact-PID/container scoped; no `pkill -f`, `killall`, broad kills.
Keep functional, proof/evidence, review, and memory commits separate.
Browser + Docker evidence required for browser/product usability claims.
Delegate concrete implementation/validation work to executor subagents where possible.
Run clean-room adversarial review before yielding.
Do not copy secrets into committed artifacts; exclude cookies, auth headers, credential values, private file contents, bulky logs.
Do not expand scope to `.msg-copy-btn`, general touch-target audits, modal focus, or range copy.
Monaco remains required; no plain textarea/diff fallback certification.
