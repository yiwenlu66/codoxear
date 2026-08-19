#!/usr/bin/env bash
# Docker-isolated behavioral verifier for the webui Pi model switch.
#
# User-facing contract exercised through the real interface: after switching
# the model from the browser (composer `/model <id>` send), the session's
# busy/idle indicator returns to idle. The container bootstrap deliberately
# leaves the Pi session with an errored tail (the offline bootstrap turn
# fails against the provider), which is the tail shape that wedged the send
# busy-latch: Pi's model_change row grew the log without opening a turn.
#
# The application is always built from a Git archive and runs under a
# container-only HOME. The host browser may connect only to a loopback port
# other than 8743; no host Codoxear runtime or deployment state is mounted.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_model_switch_busy.sh [commit]

Build (or reuse) an isolated committed Codoxear snapshot, create an offline Pi
session with an errored tail, send /model <id> through the browser composer,
and verify the busy indicator returns to idle and stays there. Results are
written to a new /tmp/codoxear-docker-model-switch-results.* directory.

Environment:
  CODOXEAR_MODEL_SWITCH_PORT       Loopback port (default: 19673; never 8743)
  CODOXEAR_MODEL_SWITCH_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_MODEL_SWITCH_PASSWORD   Login password (default: docker-model-switch-password)
USAGE
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_ref="${1:-HEAD}"
target_commit="$(git -C "$repo_root" rev-parse --verify "${target_ref}^{commit}")" || {
  echo "unable to resolve commit: $target_ref" >&2
  exit 2
}
short_commit="$(git -C "$repo_root" rev-parse --short=12 "$target_commit")"
port="${CODOXEAR_MODEL_SWITCH_PORT:-19673}"
password="${CODOXEAR_MODEL_SWITCH_PASSWORD:-docker-model-switch-password}"
image="${CODOXEAR_MODEL_SWITCH_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-model-switch-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-docker-model-switch.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-docker-model-switch-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-model-switch-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_MODEL_SWITCH_PORT must be a TCP port, got: $port" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override; unset CODEXEAR_APP_DIR and CODEX_WEB_APP_DIR" >&2
  exit 2
fi
if [[ ! -r "$source_pi_agent/models.json" || ! -r "$source_pi_agent/settings.json" ]]; then
  echo "Pi models.json and settings.json are required under $source_pi_agent" >&2
  exit 2
fi
if ! command -v agent-browser >/dev/null; then
  echo "agent-browser is required" >&2
  exit 2
fi

browser() {
  AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"
}

capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\n" "$HOME"
    find "$HOME/.pi/agent/sessions" -type f -name "*.jsonl" -printf "%p\n" 2>/dev/null
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -printf "%p\n" 2>/dev/null
    cat "$HOME/.local/share/codoxear/socks/"*.json 2>/dev/null
  ' > "$artifacts/isolation-and-session.txt" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/pi-bootstrap.log" "$HOME/broker.log"' > "$artifacts/pi-and-broker.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'tail -n 6 "$HOME/.pi/agent/sessions/"*/*.jsonl 2>/dev/null' > "$artifacts/pi-log-tail.txt" 2>&1 || true
}

cleanup() {
  capture_container_diagnostics
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

copy_pi_config() {
  umask 077
  mkdir -p "$home_dir/.pi/agent" "$home_dir/.local/share"
  local name
  for name in models.json models-store.json auth.json settings.json trust.json anthropic-context-management.json; do
    if [[ -r "$source_pi_agent/$name" ]]; then
      install -m 600 "$source_pi_agent/$name" "$home_dir/.pi/agent/$name"
    fi
  done
}

json_password() {
  python3 -c 'import json, sys; print(json.dumps({"password": sys.argv[1]}))' "$password"
}

wait_for_server() {
  for _ in $(seq 1 120); do
    if [[ "$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)" == "401" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_pi_session() {
  local sessions="$artifacts/sessions-before.json"
  for _ in $(seq 1 120); do
    local status
    status="$(curl -sS -b "$root/cookies.txt" -o "$sessions" -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions" 2>/dev/null || true)"
    if [[ "$status" == "200" ]] && python3 - "$sessions" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
sessions = payload.get("sessions") if isinstance(payload, dict) else []
raise SystemExit(0 if any(isinstance(row, dict) and row.get("agent_backend") == "pi" for row in sessions) else 1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

pi_session_field() {
  local field="$1"
  python3 - "$artifacts/sessions-live.json" "$field" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
for row in rows if isinstance(rows, list) else []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi":
        value = row.get(sys.argv[2])
        print(value if isinstance(value, (str, bool)) else "")
        break
PY
}

# The bootstrap turn fails offline, leaving an errored log tail; wait for the
# broker's error probe to settle the session to displayed idle.
wait_for_idle() {
  for _ in $(seq 1 60); do
    curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-live.json" 2>/dev/null || true
    if python3 - "$artifacts/sessions-live.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
for row in rows if isinstance(rows, list) else []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi":
        raise SystemExit(0 if not row.get("busy") and not row.get("queue_len") else 1)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

copy_pi_config

build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"
if "${docker[@]}" image inspect "$image" >/dev/null 2>&1; then
  printf 'reused image %s\n' "$image" > "$artifacts/docker-image.txt"
else
  "${docker[@]}" build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
    > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed; see $artifacts/docker-build.log"
  printf 'built image %s\n' "$image" > "$artifacts/docker-image.txt"
fi

"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach \
  --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DEFAULT_AGENT_BACKEND=pi \
  --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id docker-model-switch-session -p "Initialize isolated model-switch session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-model-switch-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$session_log" >"$HOME/broker.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\"session_id\"[[:space:]]*:[[:space:]]*\"[^\"]+\"" "$meta"; then break; fi
      sleep 1
    done
    test -n "${meta:-}" && grep -Eq "\"session_id\"[[:space:]]*:[[:space:]]*\"[^\"]+\"" "$meta"
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "isolated server did not become reachable"
curl -sS -c "$root/cookies.txt" -H 'Content-Type: application/json' --data "$(json_password)" \
  "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json" || fail "API login failed"
wait_for_pi_session || fail "offline Pi session did not appear"
wait_for_idle || fail "Pi session never went idle after the failed bootstrap turn"

curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-live.json" || fail "could not read sessions"
session_id="$(pi_session_field session_id)"
current_model="$(pi_session_field model)"
[[ -n "$session_id" ]] || fail "could not read Pi session id"
printf '%s\n' "$session_id" > "$artifacts/session-id.txt"
printf '%s\n' "$current_model" > "$artifacts/current-model.txt"

# Pick a configured provider/model different from the session's current
# model, in the same `provider/model` form the webui model picker sends. The
# container runs the same copied models.json, so any host entry is valid.
target_model="$(python3 - "$source_pi_agent/models.json" "$current_model" <<'PY'
import json
import sys
models = json.load(open(sys.argv[1], encoding="utf-8"))
current = sys.argv[2]
for provider_name, provider in (models.get("providers") or {}).items():
    for model in provider.get("models") or []:
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id.strip() and model_id != current:
            print(f"{provider_name}/{model_id}")
            raise SystemExit(0)
raise SystemExit(1)
PY
)" || fail "no alternative model available in models.json"
printf '%s\n' "$target_model" > "$artifacts/target-model.txt"

browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open isolated application"
browser fill '#pw' "$password" > "$artifacts/browser-login-fill.txt" 2>&1 || fail "browser login form did not render"
browser click '#loginBtn' > "$artifacts/browser-login-click.txt" 2>&1 || fail "browser login click failed"
browser wait 2500 > "$artifacts/browser-login-settle.txt" 2>&1 || fail "browser did not settle after login"
browser errors --clear --json > "$artifacts/browser-errors-before.json" 2>&1 || fail "could not clear browser errors"

# SEND /model: driven through the composer as the model picker would.
browser fill '#msg' "/model ${target_model}" > "$artifacts/model-fill.txt" 2>&1 || fail "composer was not available for /model"
browser wait 300 > "$artifacts/model-fill-wait.txt" 2>&1 || fail "browser wait after /model fill failed"
browser press Escape > "$artifacts/model-escape.txt" 2>&1 || fail "Escape could not close the command picker"
browser click '#sendBtn' > "$artifacts/model-send-click.txt" 2>&1 || fail "send button was not available for /model"
browser wait 800 > "$artifacts/model-send-wait.txt" 2>&1 || fail "browser wait after send failed"
browser eval '(() => { const choice = document.querySelector("#sendChoice"); return { composerValue: document.querySelector("#msg")?.value ?? null, toast: document.querySelector("#toast")?.textContent ?? null, sendChoiceOpen: Boolean(choice && getComputedStyle(choice).display === "flex") }; })()' --json > "$artifacts/model-send-state.json" 2>&1 || fail "could not observe post-send state"

# Pi must have processed the command: the session log gains a model_change row
# naming the target model, with no turn rows after it.
model_applied=""
for _ in $(seq 1 15); do
  if "${docker[@]}" exec "$container" sh -lc "grep -h '\"type\":\"model_change\"' \"\$HOME/.pi/agent/sessions/\"*/*.jsonl 2>/dev/null | grep -c '\"${target_model##*/}\"' || true" | grep -qE '^[1-9]'; then
    model_applied="1"
    break
  fi
  sleep 1
done
[[ "$model_applied" == "1" ]] || fail "Pi never recorded the model_change row; see $artifacts/pi-log-tail.txt"

# CORE CONTRACT: the busy indicator must return to idle and stay there. The
# latch quiet window is 3s; poll well past it and then re-check after an
# additional settle to catch a clear-then-restuck regression.
idle_ok=""
for i in $(seq 1 30); do
  curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-live.json" 2>/dev/null || true
  if [[ "$(pi_session_field busy)" == "False" ]]; then
    idle_ok="1"
    printf 'idle observed after %s poll(s)\n' "$i" > "$artifacts/idle-timing.txt"
    break
  fi
  sleep 1
done
[[ "$idle_ok" == "1" ]] || fail "session stayed busy after the /model send; see $artifacts/sessions-live.json"

sleep 5
curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-settled.json" 2>/dev/null || true
cp "$artifacts/sessions-settled.json" "$artifacts/sessions-live.json"
[[ "$(pi_session_field busy)" == "False" ]] || fail "session went busy again after settling; see $artifacts/sessions-settled.json"

# USER-VISIBLE INDICATOR: the sidebar state dot must render idle and the
# interrupt button must stay hidden.
browser wait 1500 > "$artifacts/browser-prewait.txt" 2>&1 || true
browser eval '(() => {
  const dots = [...document.querySelectorAll(".session .stateDot")].map((node) => node.className);
  const interrupt = document.querySelector("#interruptBtn");
  return {
    dots,
    interruptVisible: Boolean(interrupt && getComputedStyle(interrupt).display !== "none"),
  };
})()' --json > "$artifacts/indicator.json" 2>&1 || fail "could not observe the sidebar state dot"
python3 - "$artifacts/indicator.json" <<'PY' || fail "sidebar state dot did not render idle; see $artifacts/indicator.json"
import json
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))["data"]["result"]
dots = result.get("dots") or []
if not dots or not any("idle" in str(dot).split() for dot in dots):
    raise SystemExit(1)
if any("busy" in str(dot).split() for dot in dots):
    raise SystemExit(1)
if result.get("interruptVisible"):
    raise SystemExit(1)
PY

browser screenshot --full "$artifacts/verification.png" > "$artifacts/browser-screenshot.txt" 2>&1 || true

printf 'PASS: /model send settled to idle; indicator idle; results in %s\n' "$artifacts"
