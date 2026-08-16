#!/usr/bin/env bash
# Docker-isolated behavioral verifier for Pi /new from the browser.
#
# Two user-facing contracts are exercised through the real interface:
#   1. The composer's slash completion advertises /new for Pi sessions.
#   2. After /new is sent and the broker rebinds to the new session log, the
#      rendered transcript is replaced: old rows disappear instead of the new
#      session's messages appending below the dead transcript.
#
# The application is always built from a Git archive and runs under a
# container-only HOME. The host browser may connect only to a loopback port
# other than 8743; no host Codoxear runtime or deployment state is mounted.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_new_flow.sh [commit]

Build (or reuse) an isolated committed Codoxear snapshot, create an offline Pi
session, send /new through the browser composer, and verify the rendered
transcript is replaced after the log rebind. Results are written to a new
/tmp/codoxear-docker-new-flow-results.* directory.

Environment:
  CODOXEAR_NEW_FLOW_PORT       Loopback port (default: 19663; never 8743)
  CODOXEAR_NEW_FLOW_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_NEW_FLOW_PASSWORD   Login password (default: docker-new-flow-password)
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
port="${CODOXEAR_NEW_FLOW_PORT:-19663}"
password="${CODOXEAR_NEW_FLOW_PASSWORD:-docker-new-flow-password}"
image="${CODOXEAR_NEW_FLOW_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-new-flow-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-docker-new-flow.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-docker-new-flow-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-new-flow-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_NEW_FLOW_PORT must be a TCP port, got: $port" >&2
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
        print(value if isinstance(value, str) else "")
        break
PY
}

transcript_probe='
(() => {
  const inner = document.querySelector("#chatInner");
  const rows = inner ? inner.querySelectorAll(".msg-row").length : -1;
  const text = inner ? String(inner.textContent || "") : "";
  return { rows, hasOldText: text.includes("Initialize isolated new-flow session"), textSample: text.slice(0, 400) };
})()'

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
      --session-id docker-new-flow-session -p "Initialize isolated new-flow session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-new-flow-session.jsonl" -print -quit)"
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

# The offline bootstrap turn fails against the provider; the broker can seed a
# busy turn state from that log until its error probe resolves. A user would
# wait for the session to go idle before starting a new one, and an idle send
# is the path this flow verifies (a busy send opens the send-choice dialog
# instead of submitting directly).
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
    break
  fi
  sleep 1
done
python3 - "$artifacts/sessions-live.json" <<'PY' || fail "Pi session never went idle after the failed bootstrap turn"
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
for row in rows if isinstance(rows, list) else []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi":
        raise SystemExit(0 if not row.get("busy") and not row.get("queue_len") else 1)
raise SystemExit(1)
PY

curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-live.json" || fail "could not read sessions"
old_log_path="$(pi_session_field log_path)"
session_id="$(pi_session_field session_id)"
[[ -n "$old_log_path" && -n "$session_id" ]] || fail "could not read Pi session id/log_path"
printf '%s\n' "$session_id" > "$artifacts/session-id.txt"
printf '%s\n' "$old_log_path" > "$artifacts/old-log-path.txt"

browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open isolated application"
browser fill '#pw' "$password" > "$artifacts/browser-login-fill.txt" 2>&1 || fail "browser login form did not render"
browser click '#loginBtn' > "$artifacts/browser-login-click.txt" 2>&1 || fail "browser login click failed"
browser wait 2500 > "$artifacts/browser-login-settle.txt" 2>&1 || fail "browser did not settle after login"
browser errors --clear --json > "$artifacts/browser-errors-before.json" 2>&1 || fail "could not clear browser errors"

# The bootstrap turn's user message must be rendered before /new: replacing an
# already-empty transcript would prove nothing.
initial_rows=""
for _ in $(seq 1 30); do
  if browser eval "$transcript_probe" --json > "$artifacts/transcript-before.json" 2>&1; then
    initial_rows="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["result"]["rows"])' "$artifacts/transcript-before.json" 2>/dev/null || true)"
    if [[ -n "$initial_rows" && "$initial_rows" -ge 1 ]]; then break; fi
  fi
  sleep 1
done
[[ -n "$initial_rows" && "$initial_rows" -ge 1 ]] || fail "bootstrap transcript never rendered rows; see $artifacts/transcript-before.json"
printf '%s\n' "$initial_rows" > "$artifacts/initial-rows.txt"

# SLASH MENU: typing / opens the command picker; /new must be advertised.
browser fill '#msg' '/' > "$artifacts/slash-fill.txt" 2>&1 || fail "composer was not available for slash completion"
browser wait 400 > "$artifacts/slash-wait.txt" 2>&1 || fail "browser wait after slash failed"
browser eval '(() => { const picker = document.querySelector("#modelPicker"); const options = picker ? [...picker.querySelectorAll("[role=option]")] : []; return { pickerVisible: Boolean(picker && getComputedStyle(picker).display !== "none"), optionTexts: options.map((node) => node.textContent) }; })()' --json > "$artifacts/slash-picker.json" 2>&1 || fail "could not observe slash command picker"
python3 - "$artifacts/slash-picker.json" <<'PY' || fail "/new missing from the slash command picker; see $artifacts/slash-picker.json"
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
texts = payload["data"]["result"]["optionTexts"]
raise SystemExit(0 if any(str(text).startswith("/new") for text in texts) else 1)
PY
browser press Escape > "$artifacts/slash-escape.txt" 2>&1 || fail "Escape could not close the slash picker"
browser fill '#msg' '' > "$artifacts/slash-clear.txt" 2>&1 || fail "could not clear the composer"

# SEND /new: driven through the composer as a user would. Escape first closes
# the command picker so the send click is not intercepted by picker state.
browser fill '#msg' '/new' > "$artifacts/new-fill.txt" 2>&1 || fail "composer was not available for /new"
browser wait 300 > "$artifacts/new-fill-wait.txt" 2>&1 || fail "browser wait after /new fill failed"
browser press Escape > "$artifacts/new-escape.txt" 2>&1 || fail "Escape could not close the command picker"
browser eval 'document.querySelector("#msg")?.value ?? null' --json > "$artifacts/new-composer-value.json" 2>&1 || fail "could not read composer before send"
browser click '#sendBtn' > "$artifacts/new-send-click.txt" 2>&1 || fail "send button was not available for /new"
browser wait 800 > "$artifacts/new-send-wait.txt" 2>&1 || fail "browser wait after send failed"
# The composer clears on a confirmed send; the toast and the send-choice
# dialog state distinguish "submitted" from "gated behind the busy dialog".
browser eval '(() => { const choice = document.querySelector("#sendChoice"); return { composerValue: document.querySelector("#msg")?.value ?? null, toast: document.querySelector("#toast")?.textContent ?? null, sendChoiceOpen: Boolean(choice && getComputedStyle(choice).display === "flex") }; })()' --json > "$artifacts/new-send-state.json" 2>&1 || fail "could not observe post-send state"

# REBIND: Pi executes /new immediately, but a brand-new session's log file is
# only materialized on its first persisted entry, so the broker rebinds when
# the new session receives its first message. Verify Pi executed /new from
# the TUI's own confirmation, then drive the first message through the
# composer as a user would.
new_executed=""
for _ in $(seq 1 15); do
  if "${docker[@]}" exec "$container" sh -lc 'grep -c "New session started" "$HOME/broker.log" 2>/dev/null || true' | grep -qE '^[1-9]'; then
    new_executed="1"
    break
  fi
  sleep 1
done
[[ "$new_executed" == "1" ]] || fail "Pi never confirmed the new session; see $artifacts/pi-and-broker.log"

first_message="new-session-first-message-$(date +%s)"
printf '%s\n' "$first_message" > "$artifacts/first-message.txt"
browser fill '#msg' "$first_message" > "$artifacts/first-fill.txt" 2>&1 || fail "composer was not available for the first message"
browser click '#sendBtn' > "$artifacts/first-send-click.txt" 2>&1 || fail "send button was not available for the first message"
browser wait 800 > "$artifacts/first-send-wait.txt" 2>&1 || fail "browser wait after first send failed"
browser eval '(() => { const choice = document.querySelector("#sendChoice"); return { composerValue: document.querySelector("#msg")?.value ?? null, toast: document.querySelector("#toast")?.textContent ?? null, sendChoiceOpen: Boolean(choice && getComputedStyle(choice).display === "flex") }; })()' --json > "$artifacts/first-send-state.json" 2>&1 || fail "could not observe post-send state"

# REPLACE: the broker rebinds to the materialized session log (API evidence:
# log_path changes, session_id stays) and the rendered transcript must be
# replaced — old rows gone, the new session's first message rendered.
transcript_probe="
(() => {
  const inner = document.querySelector(\"#chatInner\");
  const rows = inner ? inner.querySelectorAll(\".msg-row\").length : -1;
  const text = inner ? String(inner.textContent || \"\") : \"\";
  return { rows, hasOldText: text.includes(\"Initialize isolated new-flow session\"), hasNewText: text.includes(\"${first_message}\"), textSample: text.slice(0, 400) };
})()"
rebind_ok=""
for i in $(seq 1 30); do
  curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-live.json" 2>/dev/null || true
  new_log_path="$(pi_session_field log_path)"
  if browser eval "$transcript_probe" --json > "$artifacts/transcript-after.json" 2>&1; then
    probe="$(python3 -c '
import json, sys
result = json.load(open(sys.argv[1]))["data"]["result"]
print("1" if result["rows"] >= 1 and not result["hasOldText"] and result["hasNewText"] else "0")
' "$artifacts/transcript-after.json" 2>/dev/null || true)"
    if [[ -n "$new_log_path" && "$new_log_path" != "$old_log_path" && "$probe" == "1" ]]; then
      rebind_ok="1"
      printf '%s\n' "$new_log_path" > "$artifacts/new-log-path.txt"
      printf 'rebind-and-replace observed after %s poll(s)\n' "$i" > "$artifacts/rebind-timing.txt"
      break
    fi
  fi
  sleep 1
done
[[ "$rebind_ok" == "1" ]] || fail "transcript was not replaced after /new rebind; see $artifacts/transcript-after.json and $artifacts/sessions-live.json"

# MENU AFTER REBIND: the /new session_start lets the bridge registration
# succeed, so the caps file carries the live extension registry. The projected
# menu must keep the builtins in that state (union, not replacement) and the
# bridge-provided /effort.
curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-after.json" || fail "could not read sessions after rebind"
python3 - "$artifacts/sessions-after.json" <<'PY' || fail "slash menu lost builtins after the rebind; see $artifacts/sessions-after.json"
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
row = next((r for r in rows if isinstance(r, dict) and r.get("agent_backend") == "pi"), {})
names = {str(c.get("name") or "") for c in row.get("slash_commands") or [] if isinstance(c, dict)}
missing = {"model", "new", "compact", "effort"} - names
raise SystemExit(0 if not missing else f"missing commands: {sorted(missing)}")
PY

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || fail "could not read browser errors"
browser screenshot --full "$artifacts/new-flow.png" > "$artifacts/browser-screenshot.txt" 2>&1 || fail "could not save browser screenshot"

python3 - "$artifacts" "$session_id" "$old_log_path" "$target_commit" "$port" <<'PY' || fail "report checks failed; see $artifacts/report.json"
import json
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])
session_id, old_log_path, commit, port = sys.argv[2:]

def read_json(name):
    return json.loads((artifacts / name).read_text(encoding="utf-8"))

sessions = read_json("sessions-live.json")
row = next((r for r in sessions.get("sessions", []) if r.get("agent_backend") == "pi"), {})
slash = read_json("slash-picker.json")["data"]["result"]
before = read_json("transcript-before.json")["data"]["result"]
after = read_json("transcript-after.json")["data"]["result"]
errors = read_json("browser-errors.json")
page_errors = errors.get("data", {}).get("errors") if isinstance(errors, dict) else None

slash_after = read_json("sessions-after.json")
row_after = next((r for r in slash_after.get("sessions", []) if r.get("agent_backend") == "pi"), {})
menu_names = {str(c.get("name") or "") for c in row_after.get("slash_commands") or [] if isinstance(c, dict)}

checks = {
    "session_id_stable": row.get("session_id") == session_id,
    "log_path_changed": isinstance(row.get("log_path"), str) and row["log_path"] != old_log_path,
    "slash_menu_advertises_new": any(str(t).startswith("/new") for t in slash["optionTexts"]),
    "slash_menu_keeps_builtins_after_rebind": {"model", "new", "compact", "effort"} <= menu_names,
    "transcript_rendered_before": before["rows"] >= 1 and before["hasOldText"],
    "transcript_replaced_after": after["rows"] >= 1 and not after["hasOldText"] and after["hasNewText"],
    "no_page_errors": page_errors == [],
}
report = {
    "commit": commit,
    "url": f"http://127.0.0.1:{port}/",
    "session_id": session_id,
    "old_log_path": old_log_path,
    "new_log_path": row.get("log_path"),
    "slash_options": slash["optionTexts"],
    "slash_commands_after_rebind": sorted(menu_names),
    "transcript_before": before,
    "transcript_after": after,
    "page_errors": page_errors,
    "checks": checks,
    "pass": all(checks.values()),
}
(artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["pass"] else 1)
PY

printf 'PASS: commit=%s url=http://127.0.0.1:%s/ artifacts=%s screenshot=%s\n' \
  "$target_commit" "$port" "$artifacts" "$artifacts/new-flow.png"
