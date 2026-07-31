# Upload/attachment certification — environment setup (Docker, port 19144)

Repo served: /home/yiwen/codex-web-product-recovery @ HEAD 1628498bad141b5aa841a453bb54d4b0eaa26a44 (read-only bind /workspace).
Base image: codoxear-sandbox:latest (python:3.13-slim + tmux/git/curl/ffmpeg).  cert image: codoxear-upload-cert:latest.

Cert image additions over base sandbox:
- Node 22 via deb.nodesource setup_22.x (host pi's undici requires markAsUncloneable, absent in Debian node20).
- Host pi CLI module copied in: /usr/lib/node_modules/@earendil-works (157MB, bundled node_modules),
  symlink /usr/local/bin/pi -> .../pi-coding-agent/dist/cli.js. pi 0.78.1.
- Container HOME=/home/tester (throwaway, bind-mounted from host /tmp/codoxear-upload-cert-19144/home).

Throwaway home /home/tester prepared with:
- .pi/agent/auth.json  (chmod 600; provider openai-codex OAuth token; access token valid >=2026-07-10; never printed)
- .pi/agent/models.json
- .pi/agent/settings.json stripped to {defaultProvider,defaultModel,defaultThinkingLevel} to avoid startup package reinstalls
- .pi/agent/npm copied from host (75MB; extensions incl lsp-pi + deps)
- lsp-pi sandbox repair applied: added "./node.js" export mapping (== "./node") to
  .pi/agent/npm/node_modules/vscode-languageserver-protocol/package.json
  (root cause: lsp-pi@1.0.5 imports vscode-languageserver-protocol/node.js; v3.18.x exports ./node not ./node.js).
  -> Identical to the repair used in prior Pi certification (OPS 2026-07-04T05:56:00Z).

Provider/model used: openai-codex / gpt-5.4-mini (same healthy path as prior composed Pi cert).

Feasibility gate (in-container, BEFORE browser flow):
- pi --provider openai-codex --model gpt-5.4-mini -ne -p "Reply with exactly: PING-OK"  -> PING-OK
- pi --provider openai-codex --model gpt-5.4-mini    -p "..." (extensions ON)            -> PING-OK
- pi read-tool on a staged-style app-dir absolute path /home/tester/.local/share/codoxear/uploads/<sid>/<ms>_secret.txt
  -> returned exact staged sentinel "STAGED-SENTINEL-9f3a".  Core mechanism (theorist F1) VALID: agent CAN read the app-dir staged path.

Server: docker run -d ... -p 127.0.0.1:19144:19144 -e CODEX_WEB_PASSWORD=test-password ... python3 -m codoxear.server
Isolation: APP_DIR=/home/tester/.local/share/codoxear (inside container home); pre-login /api/me 401; post-login /api/sessions 200.
Second server process for persistence proof: port 19145, same home bind, rediscovered broker-155 with pending_attachment=True from on-disk pending_attachments.json; removed after proof.

Browser: agent-browser session 'codoxear-upload-cert' (Chromium). Desktop 1280x720; mobile 390x844.

Staged uploads app-dir path (boundary record): /home/tester/.local/share/codoxear/uploads/<session_id>/<ms>_<safe_name>, mode 0600.
Real Pi read this path via its read tool (proven in desktop happy path: assistant returned exact file contents incl sentinel; backend log shows toolName "read" toolResult).

No host brokers/servers/tmux touched. No pkill/killall/broad host cleanup. Cleanup = container teardown only (docker rm -f), after artifacts saved.
Repo untouched: HEAD 1628498, no staged files, no tracked diff.
