# OPS — product experience feedback

## 2026-07-22T11:10:39+08:00 — Task initialized
- User is beginning continuous hands-on feedback against the running Docker preview.
- Tracker opened at `.memory/tasks/2026-07-22-product-experience-feedback/PROMPT.md` (Task specifications §Issues).
- Prior deployment task (`2026-07-16-docker-product-preview-feedback`) completed: preview live, cookie isolation fixed.
- Preview confirmed reachable: container `codoxear-preview-19580` healthy; tailnet HTTPS `19581` returns 401 unauthenticated.

## 2026-07-22 — Provider porting for testbed
- User requested all `~/.pi/agent/models.json` providers ported to all three backends; Codex needs OAuth too.
- models.json has three dexgem providers sharing one API key (`!cat ~/.pi/agent/.memory/local/dexgem-api-key`): `dexgem-responses` (OpenAI Responses /v1, gpt-5.6-sol/terra/luna/gpt-5.5), `dexgem-messages` (Anthropic, claude-opus/fable/kimi-k3), `dexgem-completions` (OpenAI Chat /v1, glm-5.2/deepseek-v4-pro/flash).
- Commit `c3462b4` modifies `scripts/codoxear-docker-preview` to: copy the key file into preview HOME; export `DEXGEM_API_KEY`; override `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` to dexgem-messages; generate Codex config.toml with `dexgem-responses` (wire_api=responses) and `dexgem-completions` (wire_api=chat) model_providers; preserve Codex `chatgpt` OAuth via copied auth.json.
- API verification after restart: Pi providers = `dexgem-responses/messages/completions + openai-codex`; Codex providers = `chatgpt, openai-api, dexgem-responses, dexgem-completions`; Claude env = `ANTHROPIC_BASE_URL=https://litellm.dex-gem.ai` + token set.
- Two prior user comments (reasoning efforts + cwd combobox) queued as issues #1 and #2 in this tracker, pending the provider setup that is now complete.
