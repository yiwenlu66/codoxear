#!/usr/bin/env bash
# Docker-isolated real API+UI verification for the activity peek line.
#
# Scenario, all through the real server and the real browser:
#   1. open turn with an in-flight `bash: sleep 30` call -> tail payload and
#      the busy bubble both show the peek
#   2. toolResult appended -> peek clears
#   3. second call appended -> peek replaced with `read: app.css`
#   4. result + final text appended and the fixture goes idle -> bubble gone
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_ACTIVITY_PEEK_PORT:-19672}"
password="${CODOXEAR_ACTIVITY_PEEK_PASSWORD:-activity-peek-password}"
image="codoxear-activity-peek:working-tree"
container="codoxear-activity-peek-${port}"
root="$(mktemp -d /tmp/codoxear-activity-peek-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-activity-peek-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-activity-peek-$$"
readonly repo_root port password image container root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override" >&2
  exit 2
fi
if ! command -v agent-browser >/dev/null; then
  echo "agent-browser is required" >&2
  exit 2
fi

browser() {
  AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"
}

cleanup() {
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  browser screenshot "$artifacts/fail.png" >/dev/null 2>&1 || true
  echo "artifacts=$artifacts" >&2
  exit 1
}

# ---- Build context from the WORKING TREE (tracked files at their current
# working-tree state + untracked non-ignored files, minus local-only dirs),
# then rebuild the browser bundle INTO THE CONTEXT ONLY (same esbuild
# invocation as scripts/deploy.sh). The checkout stays untouched and the
# container always serves the working tree's ESM source, never the possibly
# stale tracked dist/app.bundle.js.
build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" ls-files -z --cached --others --exclude-standard | python3 -c '
import sys
skip = (b".memory/", b"review/")
parts = sys.stdin.buffer.read().split(b"\0")
keep = [p for p in parts if p and not p.startswith(skip)]
sys.stdout.buffer.write(b"\0".join(keep) + b"\0")
' > "$root/context-files.nul"
tar --null --no-recursion -C "$repo_root" -T "$root/context-files.nul" -cf "$root/context.tar"
tar -C "$build_context" -xf "$root/context.tar"
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"

mkdir -p "$home_dir/.local/share"
sudo chown -R 1000:1000 "$home_dir"
"${docker[@]}" build --file "$build_context/docker/sandbox.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed"
"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach \
  --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.1 \
  --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$build_context,dst=/workspace,readonly" \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" \
  bash -c 'python3 tests/fixtures/activity_peek_broker.py >"$HOME/fixture.log" 2>&1 & exec python3 -m codoxear.server' \
  > "$artifacts/container-id.txt" || fail "container start failed"

for _ in $(seq 1 120); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
  [[ "$code" == "401" ]] && break
  sleep 0.25
done
[[ "${code:-}" == "401" ]] || fail "server readiness failed"

cookie="$root/cookies.txt"
curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
  --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/api-login.json"

# --- Step 1 (API): the tail seed carries the in-flight call. ---------------
for _ in $(seq 1 80); do
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions/activity-peek/messages/tail?limit=80" \
    > "$artifacts/api-tail.json" || true
  if python3 - "$artifacts/api-tail.json" <<'PY'
import json, sys
try:
    body = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
activity = body.get("activity")
raise SystemExit(0 if activity == {"label": "bash: sleep 30", "running": True} else 1)
PY
  then
    break
  fi
  sleep 0.25
done
python3 - "$artifacts/api-tail.json" <<'PY' || fail "tail payload did not carry the in-flight peek"
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
assert body.get("activity") == {"label": "bash: sleep 30", "running": True}, body.get("activity")
PY

# --- Browser: log in, select the session. -----------------------------------
browser set viewport 1280 800 >/dev/null 2>&1 || fail "viewport failed"
browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser open failed"
browser errors --clear --json >/dev/null 2>&1 || true
browser console --clear --json >/dev/null 2>&1 || true
browser fill '#pw' "$password" >/dev/null 2>&1 || fail "login fill failed"
browser click '#loginBtn' >/dev/null 2>&1 || fail "login click failed"
browser wait 2500 >/dev/null 2>&1
browser click '#sessions .session' >/dev/null 2>&1 || fail "session selection failed"
browser wait 3000 >/dev/null 2>&1

PROBE='(() => {
  const typing = document.querySelector(".msg.typing");
  const line = document.querySelector(".msg.typing .activityPeekLine");
  const stats = document.querySelector(".msg.typing .typingStats");
  return {
    typing: Boolean(typing),
    peek: line ? line.textContent : null,
    peekEmpty: line ? line.textContent === "" : null,
    stats: stats ? stats.textContent : null,
  };
})()'

append_rows() {
  # $1: JSON-lines text appended to the fixture log inside the container.
  printf '%s\n' "$1" | "${docker[@]}" exec -i "$container" sh -c 'cat >> "$HOME/activity-peek.jsonl"'
}

probe_until() {
  # $1: name, $2: python assertion expression over the probe dict as `p`.
  local name="$1" expr="$2" out=""
  for _ in $(seq 1 40); do
    browser eval "$PROBE" --json > "$artifacts/probe-${name}.json" 2>&1 || true
    if python3 - "$artifacts/probe-${name}.json" "$expr" <<'PY'
import json, sys
try:
    raw = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if not isinstance(raw, dict) or raw.get("success") is not True:
    raise SystemExit(1)
p = raw.get("data", {}).get("result")
if not isinstance(p, dict):
    raise SystemExit(1)
raise SystemExit(0 if eval(sys.argv[2], {"p": p}) else 1)
PY
    then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

# --- Step 1 (UI): the busy bubble shows the running call. -------------------
probe_until step1 'p["typing"] and p["peek"] == "bash: sleep 30…"' \
  || fail "busy bubble did not show the running peek"
browser screenshot "$artifacts/step1-peek.png" >/dev/null 2>&1 || true

# --- Step 2: the tool result clears the peek. -------------------------------
append_rows '{"type":"message","id":"r1","parentId":"a1","timestamp":"2026-10-21T06:00:30.000Z","message":{"role":"toolResult","toolCallId":"tool-1","content":[{"type":"text","text":"done"}]}}'
probe_until step2 'p["typing"] and (p["peek"] is None or p["peekEmpty"])' \
  || fail "tool result did not clear the peek"
browser screenshot "$artifacts/step2-cleared.png" >/dev/null 2>&1 || true

# --- Step 3: a second call replaces the line. -------------------------------
append_rows '{"type":"message","id":"a2","parentId":"r1","timestamp":"2026-10-21T06:00:31.000Z","message":{"role":"assistant","model":"provider/parent-model","stopReason":"toolUse","content":[{"type":"toolCall","id":"tool-2","name":"read","arguments":{"path":"app.css"}}]}}'
probe_until step3 'p["typing"] and p["peek"] == "read: app.css…"' \
  || fail "second call did not replace the peek"
browser screenshot "$artifacts/step3-replaced.png" >/dev/null 2>&1 || true

# --- Step 4: turn close removes the bubble. ---------------------------------
append_rows '{"type":"message","id":"r2","parentId":"a2","timestamp":"2026-10-21T06:00:32.000Z","message":{"role":"toolResult","toolCallId":"tool-2","content":[{"type":"text","text":"body{}"}]}}'
append_rows '{"type":"message","id":"a3","parentId":"r2","timestamp":"2026-10-21T06:00:33.000Z","message":{"role":"assistant","model":"provider/parent-model","stopReason":"stop","content":[{"type":"text","text":"All done."}]}}'
"${docker[@]}" exec "$container" touch /home/tester/activity-peek-idle
probe_until step4 'not p["typing"]' \
  || fail "typing bubble did not disappear after turn close"
browser screenshot "$artifacts/step4-idle.png" >/dev/null 2>&1 || true

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true
python3 - "$artifacts/browser-errors.json" <<'PY' || fail "browser errors recorded"
import json, sys
try:
    errors = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if isinstance(errors, dict):
    errors = errors.get("data", {}).get("errors", errors.get("errors", []))
assert not errors, errors
PY

"${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
echo "PASS: activity peek verified end to end"
echo "artifacts=$artifacts"
