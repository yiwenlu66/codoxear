#!/usr/bin/env bash
# Docker-isolated behavioral verifier for the queue badge and Pi /model picker.
#
# The application is always built from a Git archive and runs under a
# container-only HOME.  The host browser may connect only to a loopback port
# other than 8743; no host Codoxear runtime or deployment state is mounted.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_ui_flows.sh [commit]

Build (or reuse) an isolated committed Codoxear snapshot, create an offline Pi
session, and record real queue/model-picker UI flow observations. Results are
written to a new /tmp/codoxear-docker-ui-flows-results.* directory.

Environment:
  CODOXEAR_UI_FLOWS_PORT       Loopback port (default: 19653; never 8743)
  CODOXEAR_UI_FLOWS_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_UI_FLOWS_PASSWORD   Login password (default: docker-ui-flows-password)
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
port="${CODOXEAR_UI_FLOWS_PORT:-19653}"
password="${CODOXEAR_UI_FLOWS_PASSWORD:-docker-ui-flows-password}"
image="${CODOXEAR_UI_FLOWS_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-ui-flows-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-docker-ui-flows.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-docker-ui-flows-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-ui-flows-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_UI_FLOWS_PORT must be a TCP port, got: $port" >&2
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
    printf "HOME=%s\\n" "$HOME"
    printf "APP_DIR=%s\\n" "$(python3 -c "from codoxear.util import default_app_dir; print(default_app_dir())")"
    find "$HOME/.pi/agent/sessions" -type f -name "*.jsonl" -printf "%p\\n" 2>/dev/null
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -printf "%p\\n" 2>/dev/null
  ' > "$artifacts/isolation-and-session.txt" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/pi-bootstrap.log" "$HOME/broker.log"' > "$artifacts/pi-and-broker.log" 2>&1 || true
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

api_post() {
  local path="$1"
  local body="$2"
  local output="$3"
  curl -sS -b "$root/cookies.txt" -H 'Content-Type: application/json' \
    -o "$output" -w '%{http_code}' --data "$body" "http://127.0.0.1:${port}${path}"
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

wait_for_browser_badge_sample() {
  local output="$1"
  for _ in $(seq 1 20); do
    browser eval '(() => { const badge = document.querySelector("#queueBadge"); const style = badge ? getComputedStyle(badge) : null; return { exists: Boolean(badge), text: badge ? badge.textContent : null, display: style ? style.display : null, visible: Boolean(badge && style.display !== "none" && badge.getBoundingClientRect().width > 0) }; })()' --json > "$output" 2>&1 || return 1
    sleep 0.5
  done
}

ensure_browser_app() {
  if browser eval 'document.readyState' --json > "$artifacts/browser-liveness.json" 2>&1; then
    return 0
  fi
  browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-reopen.txt" 2>&1 || return 1
  if browser fill '#pw' "$password" > "$artifacts/browser-relogin-fill.txt" 2>&1; then
    browser click '#loginBtn' > "$artifacts/browser-relogin-click.txt" 2>&1 || return 1
    browser wait 1800 > "$artifacts/browser-relogin-settle.txt" 2>&1 || return 1
  fi
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
      --session-id docker-ui-flows-session -p "Initialize isolated UI-flow session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-ui-flows-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$session_log" >"$HOME/broker.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"; then break; fi
      sleep 1
    done
    test -n "${meta:-}" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "isolated server did not become reachable"
curl -sS -c "$root/cookies.txt" -H 'Content-Type: application/json' --data "$(json_password)" \
  "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json" || fail "API login failed"
wait_for_pi_session || fail "offline Pi session did not appear"

session_id="$(python3 - "$artifacts/sessions-before.json" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("sessions", [])
for row in rows:
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and isinstance(row.get("session_id"), str):
        print(row["session_id"])
        break
else:
    raise SystemExit("no Pi session id")
PY
)" || fail "could not select Pi session"
printf '%s\n' "$session_id" > "$artifacts/session-id.txt"

browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open isolated application"
browser fill '#pw' "$password" > "$artifacts/browser-login-fill.txt" 2>&1 || fail "browser login form did not render"
browser click '#loginBtn' > "$artifacts/browser-login-click.txt" 2>&1 || fail "browser login click failed"
browser wait 2500 > "$artifacts/browser-login-settle.txt" 2>&1 || fail "browser did not settle after login"
browser errors --clear --json > "$artifacts/browser-errors-before.json" 2>&1 || fail "could not clear browser errors"
browser eval '(() => { window.__codoxearUiFlowErrors = []; window.addEventListener("error", (event) => window.__codoxearUiFlowErrors.push({ message: String(event.message || ""), stack: event.error && event.error.stack ? String(event.error.stack) : null })); return true; })()' --json > "$artifacts/browser-error-listener.json" 2>&1 || fail "could not install browser error observer"

# QUEUE: Route response is retained exactly as returned. First create a real
# in-progress Pi shell command; the queue is expressly for messages that wait
# behind a current turn. `! sleep` is interpreted by the live Pi TUI, so this
# keeps the actual broker busy without substituting a fake backend.
prequeue_payload="$(python3 -c 'import json; print(json.dumps({"text": "! sleep 18"}))')"
prequeue_status="$(api_post "/api/sessions/${session_id}/send" "$prequeue_payload" "$artifacts/prequeue-send.json")" || fail "pre-queue send API call did not complete"
printf '%s\n' "$prequeue_status" > "$artifacts/prequeue-send.status"
queue_text="docker UI flow queued message $(date +%s)"
queue_payload="$(python3 -c 'import json, sys; print(json.dumps({"text": sys.argv[1]}))' "$queue_text")"
queue_enqueue_status="$(api_post "/api/sessions/${session_id}/enqueue" "$queue_payload" "$artifacts/queue-enqueue.json")" || fail "enqueue API call did not complete"
printf '%s\n' "$queue_enqueue_status" > "$artifacts/queue-enqueue.status"
browser wait 5000 > "$artifacts/queue-enqueue-wait.txt" 2>&1 || fail "browser wait after enqueue failed"
wait_for_browser_badge_sample "$artifacts/queue-badge-after-enqueue.json" || fail "could not sample queue badge after enqueue"
curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions/${session_id}/queue" > "$artifacts/queue-after-enqueue.json" || fail "queue read after enqueue failed"

queue_item_id="$(python3 - "$artifacts/queue-after-enqueue.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
items = payload.get("items") if isinstance(payload, dict) else []
if isinstance(items, list) and items and isinstance(items[0], dict) and isinstance(items[0].get("id"), str):
    print(items[0]["id"])
PY
)"
if [[ -n "$queue_item_id" ]]; then
  delete_payload="$(python3 -c 'import json, sys; print(json.dumps({"id": sys.argv[1]}))' "$queue_item_id")"
  queue_delete_status="$(api_post "/api/sessions/${session_id}/queue/delete" "$delete_payload" "$artifacts/queue-delete.json")" || fail "queue delete API call did not complete"
  printf '%s\n' "$queue_delete_status" > "$artifacts/queue-delete.status"
else
  printf 'not-attempted: no queued item remained after enqueue\n' > "$artifacts/queue-delete.status"
  printf '{"not_attempted":"no queued item remained after enqueue"}\n' > "$artifacts/queue-delete.json"
fi
browser wait 5000 > "$artifacts/queue-delete-wait.txt" 2>&1 || fail "browser wait after delete failed"
wait_for_browser_badge_sample "$artifacts/queue-badge-after-delete.json" || fail "could not sample queue badge after delete"
curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions/${session_id}/queue" > "$artifacts/queue-after-delete.json" || fail "queue read after delete failed"

# MODEL SWITCH: The composer is driven as a user would drive it. Escape must
# close the picker without sending a command; session metadata before/after is
# captured from the same API used by the UI's session refresh.
ensure_browser_app || fail "browser could not be reopened for model flow"
curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-model-before.json" || fail "could not read model before picker"
browser fill '#msg' '/model' > "$artifacts/model-fill.txt" 2>&1 || fail "composer was not available for /model"
browser wait 250 > "$artifacts/model-open-wait.txt" 2>&1 || fail "browser wait after /model failed"
browser eval '(() => { const picker = document.querySelector("#modelPicker"); const style = picker ? getComputedStyle(picker) : null; const options = picker ? [...picker.querySelectorAll("[role=option]")] : []; return { composerValue: document.querySelector("#msg")?.value ?? null, pickerDisplay: style?.display ?? null, pickerVisible: Boolean(picker && style.display !== "none" && picker.getBoundingClientRect().height > 0), optionCount: options.length, focusedOption: options.findIndex((node) => node.classList.contains("active")), activeDescendant: document.querySelector("#msg")?.getAttribute("aria-activedescendant") ?? null, optionTexts: options.slice(0, 5).map((node) => node.textContent) }; })()' --json > "$artifacts/model-picker-open.json" 2>&1 || fail "could not observe model picker"
browser press ArrowDown > "$artifacts/model-arrow-down.txt" 2>&1 || fail "ArrowDown could not be sent to model picker"
browser wait 120 > "$artifacts/model-arrow-wait.txt" 2>&1 || fail "browser wait after ArrowDown failed"
browser eval '(() => { const picker = document.querySelector("#modelPicker"); const options = picker ? [...picker.querySelectorAll("[role=option]")] : []; return { composerValue: document.querySelector("#msg")?.value ?? null, pickerDisplay: picker ? getComputedStyle(picker).display : null, focusedOption: options.findIndex((node) => node.classList.contains("active")), activeDescendant: document.querySelector("#msg")?.getAttribute("aria-activedescendant") ?? null, activeText: options.find((node) => node.classList.contains("active"))?.textContent ?? null }; })()' --json > "$artifacts/model-picker-after-arrow.json" 2>&1 || fail "could not observe ArrowDown selection"
browser press Escape > "$artifacts/model-escape.txt" 2>&1 || fail "Escape could not be sent to model picker"
browser wait 120 > "$artifacts/model-escape-wait.txt" 2>&1 || fail "browser wait after Escape failed"
browser eval '(() => { const picker = document.querySelector("#modelPicker"); return { composerValue: document.querySelector("#msg")?.value ?? null, pickerDisplay: picker ? getComputedStyle(picker).display : null, pickerVisible: Boolean(picker && getComputedStyle(picker).display !== "none" && picker.getBoundingClientRect().height > 0), optionCount: picker ? picker.querySelectorAll("[role=option]").length : 0, activeDescendant: document.querySelector("#msg")?.getAttribute("aria-activedescendant") ?? null }; })()' --json > "$artifacts/model-picker-after-escape.json" 2>&1 || fail "could not observe model picker after Escape"
curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-model-after.json" || fail "could not read model after Escape"

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || fail "could not read browser errors"
browser eval 'window.__codoxearUiFlowErrors || []' --json > "$artifacts/browser-error-details.json" 2>&1 || fail "could not read browser error details"
browser screenshot --full "$artifacts/ui-flows.png" > "$artifacts/browser-screenshot.txt" 2>&1 || fail "could not save browser screenshot"

python3 - "$artifacts" "$session_id" "$target_commit" "$port" "$image" > "$artifacts/report.json" <<'PY'
import json
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])
session_id, commit, port, image = sys.argv[2:]

def read_json(name):
    try:
        return json.loads((artifacts / name).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{name}: {exc}"}

def browser_result(name):
    payload = read_json(name)
    if isinstance(payload, dict) and payload.get("success") is True:
        data = payload.get("data")
        if isinstance(data, dict) and "result" in data:
            return data["result"]
    return {"browser_response": payload}

def pi_model(name):
    payload = read_json(name)
    rows = payload.get("sessions") if isinstance(payload, dict) else []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("session_id") == session_id:
            return {key: row.get(key) for key in ("model_provider", "model", "reasoning_effort", "agent_backend", "queue_len", "busy")}
    return {"session_not_found": True}

def read_text(name):
    try:
        return (artifacts / name).read_text(encoding="utf-8").strip()
    except Exception as exc:
        return f"read_error: {exc}"

report = {
    "commit": commit,
    "image": image,
    "url": f"http://127.0.0.1:{port}/",
    "session_id": session_id,
    "queue": {
        "prequeue_send_http_status": read_text("prequeue-send.status"),
        "prequeue_send_response": read_json("prequeue-send.json"),
        "enqueued_text": read_text("queue-enqueue.json") and None,
        "enqueue_http_status": read_text("queue-enqueue.status"),
        "enqueue_response": read_json("queue-enqueue.json"),
        "browser_badge_after_enqueue": browser_result("queue-badge-after-enqueue.json"),
        "queue_after_enqueue": read_json("queue-after-enqueue.json"),
        "delete_http_status": read_text("queue-delete.status"),
        "delete_response": read_json("queue-delete.json"),
        "browser_badge_after_delete": browser_result("queue-badge-after-delete.json"),
        "queue_after_delete": read_json("queue-after-delete.json"),
    },
    "model_switch": {
        "session_before": pi_model("sessions-model-before.json"),
        "picker_open": browser_result("model-picker-open.json"),
        "after_arrow_down": browser_result("model-picker-after-arrow.json"),
        "after_escape": browser_result("model-picker-after-escape.json"),
        "session_after_escape": pi_model("sessions-model-after.json"),
    },
    "browser_errors": browser_result("browser-errors.json"),
    "browser_error_details": browser_result("browser-error-details.json"),
}
report["queue"]["enqueued_text"] = (artifacts / "queue-enqueue.json").exists() and "stored separately in queue request artifact"
(artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

printf 'OBSERVED: commit=%s url=http://127.0.0.1:%s/ artifacts=%s report=%s\n' \
  "$target_commit" "$port" "$artifacts" "$artifacts/report.json"
