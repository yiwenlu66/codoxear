#!/usr/bin/env bash
# Isolated browser verification for the committed Codoxear application.
#
# This script deliberately never mounts the host repository or host runtime into
# the container. It builds a Git archive for the requested commit, creates a
# fresh Pi session under a throwaway container HOME, and tests the app through
# the host's Playwright-managed Chromium via agent-browser.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_verify.sh [commit]

Builds and behaviorally verifies Codoxear in Docker. The app source is exactly
[commit] (default: HEAD). Results, including verification.png and report.json,
are retained in a newly-created /tmp/codoxear-docker-verify-results.* directory.

The harness uses only a container-mounted throwaway HOME and loopback port
19643 by default. It refuses port 8743 and rejects any app-dir override that
could point the container at live Codoxear state.

Environment:
  CODOXEAR_VERIFY_PORT       Loopback port (default: 19643; never 8743)
  CODOXEAR_VERIFY_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_VERIFY_PASSWORD   Login password (default: docker-verify-password)
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
port="${CODOXEAR_VERIFY_PORT:-19643}"
password="${CODOXEAR_VERIFY_PASSWORD:-docker-verify-password}"
image="${CODOXEAR_VERIFY_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-verify-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-docker-verify.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-docker-verify-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-verify-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_VERIFY_PORT must be a TCP port, got: $port" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override; unset CODEXEAR_APP_DIR and CODEX_WEB_APP_DIR" >&2
  exit 2
fi
if [[ ! -r "$source_pi_agent/models.json" || ! -r "$source_pi_agent/auth.json" || ! -r "$source_pi_agent/settings.json" ]]; then
  echo "Pi config is incomplete under $source_pi_agent (models.json, auth.json, and settings.json are required)" >&2
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

wait_for_server() {
  local url="http://127.0.0.1:${port}/api/me"
  local code
  for _ in $(seq 1 120); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)"
    if [[ "$code" == "401" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "container server did not become reachable at $url" >&2
  return 1
}

login_and_wait_for_pi_session() {
  local cookie="$root/cookies.txt"
  curl -fsS -c "$cookie" \
    -H 'Content-Type: application/json' \
    --data "{\"password\":\"${password}\"}" \
    "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"

  local sessions="$artifacts/sessions.json"
  local sessions_status
  for _ in $(seq 1 120); do
    sessions_status="$(curl -sS -b "$cookie" -o "$sessions" -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions" 2>/dev/null || true)"
    if [[ "$sessions_status" == "200" ]] && python3 - "$sessions" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
sessions = payload.get("sessions") if isinstance(payload, dict) else None
if not isinstance(sessions, list):
    raise SystemExit(1)
if any(isinstance(session, dict) and session.get("agent_backend") == "pi" for session in sessions):
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  echo "the container-created Pi session never appeared in /api/sessions" >&2
  return 1
}

capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\\n" "$HOME"
    printf "APP_DIR=%s\\n" "$(python3 -c "from codoxear.util import default_app_dir; print(default_app_dir())")"
    printf "CODOXEAR_MODULE=%s\\n" "$(python3 -c "import codoxear; print(codoxear.__file__)")"
    find "$HOME/.pi/agent/sessions" -type f -name "*.jsonl" -printf "%p\\n" 2>/dev/null
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -printf "%p\\n" 2>/dev/null
  ' > "$artifacts/isolation-and-session.txt" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/pi-bootstrap.log" "$HOME/sessiond.log"' > "$artifacts/sessiond.log" 2>&1 || true
}

fail() {
  echo "FAIL: $*" >&2
  capture_container_diagnostics
  exit 1
}

copy_pi_config

# Build from a Git archive, not the editable checkout. The current verifier
# Dockerfile is copied into that temporary context so `scripts/docker_verify.sh
# <commit>` can inspect an older application commit using current harness logic.
build_context="$root/build-context"
mkdir -p "$build_context"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
mkdir -p "$build_context/docker"
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"
"${docker[@]}" build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed; see $artifacts/docker-build.log"

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
  "$image" \
  bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    # Pi only materializes a native session JSONL after its first turn. Run an
    # offline initialization turn first; its expected provider failure is still
    # recorded in the Pi-owned session file, then the container broker resumes it.
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id docker-verify-session -p "Initialize isolated verification session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-verify-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$session_log" >"$HOME/sessiond.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"; then
        break
      fi
      sleep 1
    done
    test -n "${meta:-}" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "server readiness check failed"
login_and_wait_for_pi_session || fail "Pi session creation/discovery failed"

browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open app"
# The unauthenticated shell intentionally probes /api/me and receives 401. Clear
# that expected pre-login network diagnostic; the captured buffers below cover
# only the authenticated application bootstrap.
browser errors --clear --json > "$artifacts/browser-errors-before.json" 2>&1 || fail "unable to clear browser error buffer"
browser console --clear --json > "$artifacts/browser-console-before.json" 2>&1 || fail "unable to clear browser console buffer"
browser fill '#pw' "$password" > "$artifacts/browser-login-fill.txt" 2>&1 || fail "login password field was not available"
browser click '#loginBtn' > "$artifacts/browser-login-click.txt" 2>&1 || fail "login button was not available"
browser wait 3500 > "$artifacts/browser-settle.txt" 2>&1 || fail "browser did not settle after login"
browser snapshot -i -c --depth 12 > "$artifacts/browser-snapshot.txt" 2>&1 || fail "browser snapshot failed"
browser eval '(() => {
  const visible = (element) => Boolean(element) && getComputedStyle(element).display !== "none" && getComputedStyle(element).visibility !== "hidden" && element.getBoundingClientRect().width > 0 && element.getBoundingClientRect().height > 0;
  const sessions = document.querySelector("#sessions");
  const sidebar = document.querySelector(".sidebar");
  const cards = sessions ? sessions.querySelectorAll(":scope > .session").length : 0;
  const bundleLoaded = Boolean(document.querySelector('script[type="module"][src*="dist/app.bundle.js"]'));
  return {
    appBootstrapped: window.__codoxearAppBootstrapped === true,
    loadError: window.__codoxearLoadError ?? null,
    cards,
    sessionListRendered: Boolean(sessions && sessions.dataset.codoxearSessionsRendered === "true"),
    sidebarVisible: visible(sidebar),
    sidebarContent: String(sidebar?.innerText || "").trim().length,
    bundleLoaded,
    visibleLoadFailure: /codoxear failed to load|error: unable to contact server/i.test(String(document.body?.innerText || ""))
  };
})()' --json > "$artifacts/browser-report.json" 2>&1 || fail "DOM verification evaluation failed"
browser errors --json > "$artifacts/browser-errors.json" 2>&1 || fail "browser error report failed"
browser console --json > "$artifacts/browser-console.json" 2>&1 || fail "browser console report failed"
browser screenshot --full "$artifacts/verification.png" > "$artifacts/browser-screenshot.txt" 2>&1 || fail "screenshot capture failed"

if python3 - "$artifacts/browser-report.json" "$artifacts/browser-errors.json" "$artifacts/browser-console.json" "$artifacts/report.json" <<'PY'
import json
import re
import sys
from pathlib import Path

report_path, errors_path, console_path, output_path = map(Path, sys.argv[1:])

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid JSON from {path.name}: {exc}")

raw_report = load(report_path)
raw_errors = load(errors_path)
raw_console = load(console_path)

def browser_data(payload, name):
    if not isinstance(payload, dict) or payload.get("success") is not True or not isinstance(payload.get("data"), dict):
        raise SystemExit(f"agent-browser {name} returned an unsuccessful response: {payload!r}")
    return payload["data"]

report_data = browser_data(raw_report, "eval")
errors_data = browser_data(raw_errors, "errors")
console_data = browser_data(raw_console, "console")
report = report_data.get("result")
errors = errors_data.get("errors")
console = console_data.get("messages")
if not isinstance(report, dict):
    raise SystemExit(f"agent-browser eval did not return an object: {report!r}")
if not isinstance(errors, list):
    raise SystemExit(f"agent-browser errors did not return a list: {errors!r}")
if not isinstance(console, list):
    raise SystemExit(f"agent-browser console did not return a list: {console!r}")
checks = {
    "app_bootstrapped": report.get("appBootstrapped") is True,
    "no_load_error": report.get("loadError") is None,
    "session_cards_rendered": isinstance(report.get("cards"), int) and report["cards"] > 0,
    "session_list_rendered": report.get("sessionListRendered") is True,
    "sidebar_visible": report.get("sidebarVisible") is True,
    "sidebar_has_content": isinstance(report.get("sidebarContent"), int) and report["sidebarContent"] > 0,
    "bundle_loaded": report.get("bundleLoaded") is True,
    "no_visible_load_failure": report.get("visibleLoadFailure") is False,
    "no_page_errors": errors == [],
}
# Console resource diagnostics such as a missing favicon arrive as `error` entries,
# but do not mean the application failed. Keep those diagnostics in the report
# and fail only for JavaScript exceptions or Codoxear module-load failures.
error_console = [
    entry for entry in console
    if isinstance(entry, dict) and str(entry.get("type") or entry.get("level") or "").lower() == "error"
]
fatal_console_errors = [
    entry for entry in error_console
    if re.search(
        r"\b(?:ReferenceError|TypeError|SyntaxError)\b|\bCodoxear\b.*\bfailed to load\b",
        str(entry.get("text") or ""),
        re.IGNORECASE,
    )
]
checks["no_fatal_console_errors"] = not fatal_console_errors
summary = {
    "pass": all(checks.values()),
    "checks": checks,
    "browserReport": report,
    "pageErrors": errors,
    "fatalConsoleErrors": fatal_console_errors,
    "ignoredConsoleErrors": [
        entry for entry in error_console if entry not in fatal_console_errors
    ],
}
output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
if not summary["pass"]:
    raise SystemExit(1)
PY
then
  verification_status=0
else
  verification_status=1
fi

capture_container_diagnostics
if (( verification_status != 0 )); then
  echo "FAIL: browser verification failed; artifacts=$artifacts" >&2
  exit "$verification_status"
fi

printf 'PASS: commit=%s url=http://127.0.0.1:%s/ artifacts=%s screenshot=%s\n' \
  "$target_commit" "$port" "$artifacts" "$artifacts/verification.png"
