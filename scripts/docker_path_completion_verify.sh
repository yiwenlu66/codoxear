#!/usr/bin/env bash
# Docker-isolated behavioral verifier for composer filesystem path completion.
#
# The application image is built from a Git archive of a committed ref (HEAD by
# default), never the editable checkout. A throwaway container HOME holds both
# the Pi session and a deliberately small fixture tree. Host Chromium is driven
# through agent-browser against only the container's loopback-published port.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_path_completion_verify.sh [commit]

Builds [commit] (default HEAD) from a Git archive and behaviorally verifies the
composer path-completion API and UI in Docker. Artifacts, including screenshots,
API bodies, browser snapshots, and container diagnostics remain in a fresh
/tmp/codoxear-path-completion-results.* directory.

Environment:
  CODOXEAR_PATH_COMPLETION_VERIFY_PORT       Loopback port (default 19684; never 8743)
  CODOXEAR_PATH_COMPLETION_VERIFY_IMAGE      Docker image tag override
  CODOXEAR_PATH_COMPLETION_VERIFY_PASSWORD   Login password override
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
port="${CODOXEAR_PATH_COMPLETION_VERIFY_PORT:-19684}"
password="${CODOXEAR_PATH_COMPLETION_VERIFY_PASSWORD:-docker-path-completion-verify-password}"
image="${CODOXEAR_PATH_COMPLETION_VERIFY_IMAGE:-codoxear-path-completion-verify:${short_commit}}"
container="codoxear-path-completion-verify-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-path-completion-verify.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-path-completion-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-path-completion-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_PATH_COMPLETION_VERIFY_PORT must be a TCP port, got: $port" >&2
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
  "${docker[@]}" logs "$container" > "$artifacts/server-container.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\\n" "$HOME"
    printf "APP_DIR=%s\\n" "$(python3 -c "from codoxear.util import default_app_dir; print(default_app_dir())")"
    printf -- "--- fixture tree ---\\n"
    find "$HOME/path-completion-fixture" -maxdepth 3 -printf "%y %p\\n" 2>/dev/null || true
    printf -- "--- socks metadata ---\\n"
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -exec sh -c '\''echo --- "$1"; cat "$1"'\'' _ {} \; 2>/dev/null || true
    printf -- "--- Pi session logs ---\\n"
    find "$HOME/.pi/agent/sessions" -type f -name "*.jsonl" -printf "%p\\n" 2>/dev/null || true
  ' > "$artifacts/isolation-and-socks.txt" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/pi-bootstrap.log" "$HOME/broker.log"' > "$artifacts/pi-and-broker.log" 2>&1 || true
}

cleanup() {
  capture_container_diagnostics
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  "${docker[@]}" rmi "$image" >/dev/null 2>&1 || true
  /bin/rm -rf "$root"
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
  # The mention-link scenario sends one real prompt, so the container Pi needs
  # the configured API-key file dependency (the picker scenarios never did).
  # Copy only literal `!cat` reads under the agent dir; never execute commands.
  python3 - "$source_pi_agent" "$home_dir/.pi/agent" <<'PYKEY'
import json, pathlib, re, shutil, sys
source, target = map(pathlib.Path, sys.argv[1:])
models = json.loads((source / "models.json").read_text())
for provider in models.get("providers", {}).values():
    value = provider.get("apiKey", "")
    match = re.fullmatch(r"!cat ~/.pi/agent/([A-Za-z0-9._/-]+)", value) if isinstance(value, str) else None
    if not match:
        continue
    src = (source / match[1]).resolve()
    if not src.is_relative_to(source.resolve()) or not src.is_file():
        raise SystemExit("configured Pi API-key file dependency unavailable")
    dst = target / match[1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    dst.chmod(0o600)
PYKEY
  # Keep provider/model defaults, not the host's package auto-install workflow.
  python3 - "$home_dir/.pi/agent/settings.json" <<'PYSETTINGS'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
settings = json.loads(p.read_text()) if p.exists() else {}
for key in ("packages", "extensions", "skills", "promptTemplates"):
    settings[key] = []
settings["defaultProjectTrust"] = "always"
p.write_text(json.dumps(settings))
PYSETTINGS
}

wait_for_server() {
  local url="http://127.0.0.1:${port}/api/me"
  local status
  for _ in $(seq 1 120); do
    status="$(curl -sS -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)"
    [[ "$status" == "401" ]] && return 0
    sleep 1
  done
  return 1
}

api_get() { # cookie path output
  curl -sS -b "$1" -o "$3" -w '%{http_code}' "http://127.0.0.1:${port}$2"
}

api_complete() { # path output
  local raw_path="$1" output="$2"
  curl -sS -G -b "$root/cookies.txt" -o "$output" -w '%{http_code}' \
    --data-urlencode "path=$raw_path" --data-urlencode 'limit=50' \
    "http://127.0.0.1:${port}/api/sessions/${session_id}/file/complete"
}

wait_for_pi_session() { # output path
  local output="$1"
  for _ in $(seq 1 120); do
    local status
    status="$(api_get "$root/cookies.txt" /api/sessions "$output" || true)"
    if [[ "$status" == "200" ]] && python3 - "$output" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
raise SystemExit(0 if any(isinstance(row, dict) and row.get("agent_backend") == "pi" and isinstance(row.get("session_id"), str) for row in rows) else 1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

picker_eval='(() => {
  const picker = document.querySelector("#modelPicker");
  const visible = (node) => Boolean(node) && getComputedStyle(node).display !== "none" && getComputedStyle(node).visibility !== "hidden" && node.getBoundingClientRect().width > 0 && node.getBoundingClientRect().height > 0;
  const options = Array.from(picker?.querySelectorAll(".modelPickerOption") || []);
  const paths = Array.from(picker?.querySelectorAll(".pathPickerOption") || []);
  return {
    value: document.querySelector("#msg")?.value ?? null,
    caret: document.querySelector("#msg")?.selectionStart ?? null,
    pickerVisible: visible(picker),
    ariaLabel: picker?.getAttribute("aria-label") ?? null,
    header: picker?.querySelector(".modelPickerProvider")?.textContent ?? null,
    options: options.map((node) => ({ text: node.textContent, active: node.classList.contains("active") })),
    pathOptions: paths.map((node) => ({ text: node.textContent, active: node.classList.contains("active") })),
  };
})()'

capture_picker() { # artifact basename
  browser eval "$picker_eval" --json > "$artifacts/$1.json" 2>&1 || fail "browser picker snapshot failed: $1"
}

evalv() { browser eval "$1" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["result"])'; }

wait_js() { # condition attempts
  local condition="$1" attempts="${2:-80}"
  for _ in $(seq 1 "$attempts"); do
    if [[ "$(evalv "$condition" 2>/dev/null)" == "True" ]]; then return 0; fi
    sleep 0.25
  done
  return 1
}

wait_for_picker() { # artifact basename wait milliseconds
  sleep "${2:-1}"
  capture_picker "$1"
}

copy_pi_config

# Build from the immutable requested revision. app.bundle.js is deliberately
# regenerated in the throwaway archive because this feature's source commit
# precedes generated bundle updates; the editable checkout is never written.
build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"
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
  "$image" bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    fixture="$HOME/path-completion-fixture"
    mkdir -p "$fixture/subdir" "$fixture/my dir"
    : > "$fixture/plain.txt"
    : > "$fixture/.hidden"
    : > "$fixture/subdir/child.txt"
    : > "$fixture/my dir/space-child.txt"
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id docker-path-completion-verify-session -p "Initialize isolated path-completion verification session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-path-completion-verify-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd "$fixture" -- \
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
curl -fsS -c "$root/cookies.txt" -H 'Content-Type: application/json' \
  --data "$(python3 -c 'import json,sys; print(json.dumps({"password": sys.argv[1]}))' "$password")" \
  "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json" || fail "API login failed"
wait_for_pi_session "$artifacts/api-sessions.json" || fail "offline Pi session did not appear"

session_id="$(python3 - "$artifacts/api-sessions.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8")).get("sessions", []):
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and isinstance(row.get("session_id"), str):
        print(row["session_id"])
        break
else:
    raise SystemExit("no Pi session id")
PY
)" || fail "could not select Pi session"
printf '%s\n' "$session_id" > "$artifacts/session-id.txt"
card_selector="#sessions .session[data-session-id=\"${session_id}\"]"

# ============================ API PHASE ============================
printf '%s\n' "$(api_complete './' "$artifacts/api-cwd.json")" > "$artifacts/api-cwd.status"
printf '%s\n' "$(api_complete './.' "$artifacts/api-hidden.json")" > "$artifacts/api-hidden.status"
printf '%s\n' "$(api_complete './pln' "$artifacts/api-fuzzy.json")" > "$artifacts/api-fuzzy.status"
printf '%s\n' "$(api_complete '../' "$artifacts/api-parent.json")" > "$artifacts/api-parent.status"
printf '%s\n' "$(api_complete '~/' "$artifacts/api-home.json")" > "$artifacts/api-home.status"
printf '%s\n' "$(api_complete '/etc/ho' "$artifacts/api-etc-hosts.json")" > "$artifacts/api-etc-hosts.status"
printf '%s\n' "$(api_complete '' "$artifacts/api-mention-cwd.json")" > "$artifacts/api-mention-cwd.status"
printf '%s\n' "$(api_complete 'pln' "$artifacts/api-mention-fragment.json")" > "$artifacts/api-mention-fragment.status"
printf '%s\n' "$(curl -sS -b "$root/cookies.txt" -o "$artifacts/api-missing-path.json" -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions/${session_id}/file/complete?limit=50")" > "$artifacts/api-missing-path.status"
printf '%s\n' "$(curl -sS -G -b "$root/cookies.txt" -o "$artifacts/api-unknown.json" -w '%{http_code}' --data-urlencode 'path=./' "http://127.0.0.1:${port}/api/sessions/no-such-session/file/complete")" > "$artifacts/api-unknown.status"

python3 - "$artifacts" <<'PY' > "$artifacts/api-checks.json"
import json, sys
from pathlib import Path

artifacts = Path(sys.argv[1])
def body(name):
    try: return json.loads((artifacts / name).read_text(encoding="utf-8"))
    except Exception as exc: return {"read_error": str(exc)}
def status(name):
    try: return (artifacts / name).read_text(encoding="utf-8").strip()
    except Exception as exc: return f"read_error: {exc}"
def entries(name):
    value = body(name)
    return value.get("entries", []) if isinstance(value, dict) else []
def names(name): return [entry.get("name") for entry in entries(name) if isinstance(entry, dict)]
def match(name, needle): return any(entry.get("name") == needle for entry in entries(name) if isinstance(entry, dict))

cwd = body("api-cwd.json")
home = body("api-home.json")
checks = {
    "cwd_listing_shape": status("api-cwd.status") == "200" and cwd.get("ok") is True and cwd.get("dir") == "/home/tester/path-completion-fixture" and isinstance(cwd.get("truncated"), bool),
    "cwd_lists_fixture_and_hides_dotfile": {"subdir", "my dir", "plain.txt"}.issubset(set(names("api-cwd.json"))) and ".hidden" not in names("api-cwd.json"),
    "directories_have_slashed_completion": all(any(entry.get("name") == item and entry.get("is_dir") is True and entry.get("completion") == f"./{item}/" for entry in entries("api-cwd.json")) for item in ("subdir", "my dir")),
    "dot_fragment_exposes_hidden": status("api-hidden.status") == "200" and match("api-hidden.json", ".hidden"),
    "fuzzy_fragment_filters_plain_file": status("api-fuzzy.status") == "200" and names("api-fuzzy.json") == ["plain.txt"],
    "parent_listing_includes_fixture": status("api-parent.status") == "200" and body("api-parent.json").get("dir") == "/home/tester" and match("api-parent.json", "path-completion-fixture"),
    "home_listing_includes_fixture": status("api-home.status") == "200" and home.get("dir") == "/home/tester" and match("api-home.json", "path-completion-fixture"),
    "absolute_etc_fragment_matches_hosts": status("api-etc-hosts.status") == "200" and body("api-etc-hosts.json").get("dir") == "/etc" and match("api-etc-hosts.json", "hosts"),
    "unknown_session_is_404": status("api-unknown.status") == "404",
    # The `@` mention queries the cwd with an empty token and re-prefixes bare
    # completions, so the server must return them without a `./` directory part.
    "mention_empty_token_lists_cwd": status("api-mention-cwd.status") == "200" and body("api-mention-cwd.json").get("dir") == "/home/tester/path-completion-fixture" and any(entry.get("name") == "subdir" and entry.get("completion") == "subdir/" and entry.get("is_dir") is True for entry in entries("api-mention-cwd.json")),
    "mention_slashless_fragment_narrows": status("api-mention-fragment.status") == "200" and names("api-mention-fragment.json") == ["plain.txt"] and entries("api-mention-fragment.json")[0].get("completion") == "plain.txt",
    "missing_path_parameter_is_400": status("api-missing-path.status") == "400",
}
print(json.dumps({"checks": checks, "observed": {"cwd": cwd, "hidden": body("api-hidden.json"), "fuzzy": body("api-fuzzy.json"), "parent": body("api-parent.json"), "home": home, "etc": body("api-etc-hosts.json"), "unknown": body("api-unknown.json"), "mentionCwd": body("api-mention-cwd.json"), "mentionFragment": body("api-mention-fragment.json"), "missingPath": body("api-missing-path.json")}, "pass": all(checks.values())}, indent=2))
PY
python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$artifacts/api-checks.json" || fail "API phase failed; see $artifacts/api-checks.json"

# ============================ UI PHASE =============================
app_url="http://127.0.0.1:${port}/"
browser open "$app_url" > "$artifacts/ui-open.txt" 2>&1 || fail "browser could not open app"
browser errors --clear --json > "$artifacts/browser-errors-before.json" 2>&1 || true
browser console --clear --json > "$artifacts/browser-console-before.json" 2>&1 || true
browser fill '#pw' "$password" > "$artifacts/ui-login-fill.txt" 2>&1 || fail "login form did not render"
browser click '#loginBtn' > "$artifacts/ui-login-click.txt" 2>&1 || fail "login button did not render"
browser wait 3000 > "$artifacts/ui-login-settle.txt" 2>&1 || fail "browser did not settle after login"
browser click "$card_selector" > "$artifacts/ui-select-session.txt" 2>&1 || fail "session card did not render"
browser wait 1200 > /dev/null 2>&1 || true

# `./` opens a path picker whose header is the resolved session cwd.
browser focus '#msg' > "$artifacts/ui-focus-composer.txt" 2>&1 || fail "composer did not render"
browser type '#msg' './' > "$artifacts/ui-type-dot-slash.txt" 2>&1 || fail "could not type ./"
wait_for_picker ui-01-dot-slash 1
browser screenshot --full "$artifacts/ui-01-dot-slash.png" > "$artifacts/ui-01-dot-slash-screenshot.txt" 2>&1 || true

# A trailing fragment narrows the listing.
browser keyboard type 'pln' > "$artifacts/ui-type-fragment.txt" 2>&1 || fail "could not type fragment"
wait_for_picker ui-02-fragment 1

# Navigate to subdir with ArrowDown (my dir is alphabetically first), drill
# with Tab, then accept its child file with Enter.
browser fill '#msg' './' > "$artifacts/ui-reset-dot-slash.txt" 2>&1 || fail "could not reset composer"
wait_for_picker ui-03-before-drill 1
browser press ArrowDown > "$artifacts/ui-arrow-subdir.txt" 2>&1 || fail "ArrowDown failed"
capture_picker ui-04-subdir-active
browser press Tab > "$artifacts/ui-tab-drill.txt" 2>&1 || fail "Tab drill failed"
wait_for_picker ui-05-subdir-children 1
browser screenshot --full "$artifacts/ui-05-subdir-children.png" > "$artifacts/ui-05-subdir-children-screenshot.txt" 2>&1 || true
browser press Enter > "$artifacts/ui-enter-file.txt" 2>&1 || fail "Enter file acceptance failed"
wait_for_picker ui-06-file-accepted 0.4
browser screenshot --full "$artifacts/ui-06-file-accepted.png" > "$artifacts/ui-06-file-accepted-screenshot.txt" 2>&1 || true

# The accepted file is a draft. Reload and reselect through the real UI;
# browser-side/session draft restoration must retain the completed path.
browser wait 1800 > /dev/null 2>&1 || true
browser open "$app_url" > "$artifacts/ui-path-draft-reload.txt" 2>&1 || fail "reload after path acceptance failed"
browser wait 3000 > /dev/null 2>&1 || true
browser click "$card_selector" > "$artifacts/ui-path-draft-reselect.txt" 2>&1 || fail "could not reselect session after reload"
browser wait 1400 > /dev/null 2>&1 || true
capture_picker ui-07-path-draft-reloaded
browser screenshot --full "$artifacts/ui-07-path-draft-reloaded.png" > "$artifacts/ui-07-path-draft-reloaded-screenshot.txt" 2>&1 || true

# An accepted directory containing a space retains the anchored token span and
# lists its child rather than treating the space as a new ordinary token.
browser fill '#msg' './my' > "$artifacts/ui-spacey-query.txt" 2>&1 || fail "could not type spacey directory query"
wait_for_picker ui-08-spacey-option 1
browser press Tab > "$artifacts/ui-spacey-tab.txt" 2>&1 || fail "Tab spacey directory acceptance failed"
wait_for_picker ui-09-spacey-children 1
browser screenshot --full "$artifacts/ui-09-spacey-children.png" > "$artifacts/ui-09-spacey-children-screenshot.txt" 2>&1 || true

# Index-zero slash preserves the command picker; a slash after whitespace is
# an absolute-path token and therefore has a path-picker header for /.
browser fill '#msg' '/' > "$artifacts/ui-index-zero-slash.txt" 2>&1 || fail "could not type index-zero slash"
wait_for_picker ui-10-index-zero-slash 0.5
browser fill '#msg' ' /e' > "$artifacts/ui-mid-message-absolute.txt" 2>&1 || fail "could not type absolute path token"
wait_for_picker ui-11-mid-message-absolute 1
browser screenshot --full "$artifacts/ui-11-mid-message-absolute.png" > "$artifacts/ui-11-mid-message-absolute-screenshot.txt" 2>&1 || true

# URLs are ordinary text, not a path token. Esc closes a path picker, while a
# subsequent input opens it again. A physical click near the left text inset
# moves the caret out of the active token and closes the picker.
browser fill '#msg' 'https://example.com' > "$artifacts/ui-url-nontrigger.txt" 2>&1 || fail "could not type URL"
wait_for_picker ui-12-url-nontrigger 0.5
browser fill '#msg' './' > "$artifacts/ui-escape-source.txt" 2>&1 || fail "could not type escape source"
wait_for_picker ui-13-before-escape 1
browser press Escape > "$artifacts/ui-escape.txt" 2>&1 || fail "Escape failed"
wait_for_picker ui-14-after-escape 0.3
browser keyboard type 'p' > "$artifacts/ui-reopen-after-escape.txt" 2>&1 || fail "could not retype after Escape"
wait_for_picker ui-15-reopened 1
browser fill '#msg' './plain.txt' > "$artifacts/ui-caret-source.txt" 2>&1 || fail "could not type caret source"
wait_for_picker ui-16-before-caret-click 1
browser eval '(() => { const r = document.querySelector("#msg").getBoundingClientRect(); return { x: Math.round(r.left + 2), y: Math.round(r.top + r.height / 2) }; })()' --json > "$artifacts/ui-caret-click-point.json" 2>&1 || fail "could not get composer geometry"
read -r click_x click_y < <(python3 - "$artifacts/ui-caret-click-point.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
point = payload["data"]["result"]
print(point["x"], point["y"])
PY
)
browser mouse move "$click_x" "$click_y" > "$artifacts/ui-caret-click.txt" 2>&1 || fail "physical caret move failed"
browser mouse down left >> "$artifacts/ui-caret-click.txt" 2>&1 || fail "physical caret press failed"
browser mouse up left >> "$artifacts/ui-caret-click.txt" 2>&1 || fail "physical caret release failed"
wait_for_picker ui-17-after-caret-click 0.5
browser screenshot --full "$artifacts/ui-17-after-caret-click.png" > "$artifacts/ui-17-after-caret-click-screenshot.txt" 2>&1 || true
# ---------------------------------------------------------------------------
# `@` mentions: the same picker anchored at the session cwd, with the marker
# kept in the inserted text; an email local part never triggers it; a no-match
# listing must not swallow Enter; and the sent message renders a working link.
# ---------------------------------------------------------------------------
browser fill '#msg' '@' > "$artifacts/ui-mention-open.txt" 2>&1 || fail "could not type bare @"
wait_for_picker ui-18-mention-open 1
browser screenshot --full "$artifacts/ui-18-mention-open.png" > "$artifacts/ui-18-mention-open-screenshot.txt" 2>&1 || true
browser keyboard type 'pln' > "$artifacts/ui-mention-fragment.txt" 2>&1 || fail "could not type mention fragment"
wait_for_picker ui-19-mention-fragment 1
browser press Enter > "$artifacts/ui-mention-accept.txt" 2>&1 || fail "Enter mention acceptance failed"
wait_for_picker ui-20-mention-accepted 0.4
browser screenshot --full "$artifacts/ui-20-mention-accepted.png" > "$artifacts/ui-20-mention-accepted-screenshot.txt" 2>&1 || true

browser fill '#msg' 'foo@bar' > "$artifacts/ui-mention-email.txt" 2>&1 || fail "could not type email token"
wait_for_picker ui-21-mention-email 0.5

browser fill '#msg' 'mail @zzz' > "$artifacts/ui-mention-nomatch.txt" 2>&1 || fail "could not type no-match mention"
wait_for_picker ui-22-mention-nomatch 0.6
browser press Enter > "$artifacts/ui-mention-enter.txt" 2>&1 || fail "Enter on a no-match mention failed"
browser eval '(() => ({ value: document.querySelector("#msg").value }))()' --json > "$artifacts/ui-23-mention-enter.json" 2>&1 || fail "could not read composer after no-match Enter"

browser fill '#msg' '@plain.txt' > "$artifacts/ui-mention-send-source.txt" 2>&1 || fail "could not fill mention message"
wait_for_picker ui-mention-send-source 0.4
browser click '#sendBtn' > "$artifacts/ui-mention-send.txt" 2>&1 || fail "could not send mention message"
wait_js 'Boolean(document.querySelector(".msg-row.user a.inlineFileLink[data-file-path$=\"/plain.txt\"]"))' 120 || fail "mention did not render as a file link"
browser eval '(() => { const link = document.querySelector(".msg-row.user a.inlineFileLink[data-file-path$=\"/plain.txt\"]"); return { text: link ? link.textContent : null, kind: link ? link.getAttribute("data-file-kind") : null, path: link ? link.getAttribute("data-file-path") : null, pendingLink: Boolean(link && link.closest(".msg-row[data-pending]")) }; })()' --json > "$artifacts/ui-24-mention-link.json" 2>&1 || fail "could not read mention link"
browser screenshot --full "$artifacts/ui-24-mention-link.png" > "$artifacts/ui-24-mention-link-screenshot.txt" 2>&1 || true
browser eval '(() => { const link = document.querySelector(".msg-row.user a.inlineFileLink[data-file-path$=\"/plain.txt\"]"); if (!link) return false; link.click(); return true; })()' > "$artifacts/ui-mention-link-click.txt" 2>&1 || fail "mention link click failed"
wait_js 'document.querySelector("#fileViewer") && getComputedStyle(document.querySelector("#fileViewer")).display !== "none" && String(document.querySelector("#filePickerInput").value).endsWith("plain.txt")' 80 || fail "mention link did not open the viewer on plain.txt"
browser eval '(() => ({ viewerDisplay: getComputedStyle(document.querySelector("#fileViewer")).display, path: document.querySelector("#filePickerInput").value, status: document.querySelector("#fileStatus").textContent }))()' --json > "$artifacts/ui-25-mention-viewer.json" 2>&1 || fail "could not read the viewer after the mention click"
browser screenshot --full "$artifacts/ui-25-mention-viewer.png" > "$artifacts/ui-25-mention-viewer-screenshot.txt" 2>&1 || true

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true

python3 - "$artifacts" <<'PY' > "$artifacts/ui-checks.json"
import json, sys
from pathlib import Path

artifacts = Path(sys.argv[1])
def result(name):
    try:
        payload = json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
        return payload.get("data", {}).get("result", {}) if isinstance(payload, dict) else {}
    except Exception as exc:
        return {"read_error": str(exc)}
def texts(state): return [row.get("text") for row in state.get("pathOptions", []) if isinstance(row, dict)]
def active(state): return next((row.get("text") for row in state.get("pathOptions", []) if isinstance(row, dict) and row.get("active")), None)

first = result("ui-01-dot-slash")
fragment = result("ui-02-fragment")
subdir_active = result("ui-04-subdir-active")
subdir = result("ui-05-subdir-children")
file_accepted = result("ui-06-file-accepted")
draft_reload = result("ui-07-path-draft-reloaded")
space_option = result("ui-08-spacey-option")
space_children = result("ui-09-spacey-children")
index_zero = result("ui-10-index-zero-slash")
absolute = result("ui-11-mid-message-absolute")
url = result("ui-12-url-nontrigger")
before_escape = result("ui-13-before-escape")
after_escape = result("ui-14-after-escape")
reopened = result("ui-15-reopened")
before_caret = result("ui-16-before-caret-click")
after_caret = result("ui-17-after-caret-click")
mention_open = result("ui-18-mention-open")
mention_fragment = result("ui-19-mention-fragment")
mention_accepted = result("ui-20-mention-accepted")
mention_email = result("ui-21-mention-email")
mention_nomatch = result("ui-22-mention-nomatch")
mention_enter = result("ui-23-mention-enter")
mention_link = result("ui-24-mention-link")
mention_viewer = result("ui-25-mention-viewer")

checks = {
    "dot_slash_opens_fixture_picker": first.get("pickerVisible") is True and first.get("ariaLabel") == "Available paths" and first.get("header") == "/home/tester/path-completion-fixture" and {"my dir/", "subdir/", "plain.txt"}.issubset(set(texts(first))),
    "dot_slash_hides_dotfile": ".hidden" not in texts(first),
    "fragment_narrows_to_plain_file": fragment.get("value") == "./pln" and texts(fragment) == ["plain.txt"],
    "arrow_selects_subdir": active(subdir_active) == "subdir/",
    "tab_drills_subdir": subdir.get("value") == "./subdir/" and subdir.get("pickerVisible") is True and subdir.get("header") == "/home/tester/path-completion-fixture/subdir" and texts(subdir) == ["child.txt"],
    "enter_accepts_file_and_closes": file_accepted.get("value") == "./subdir/child.txt" and file_accepted.get("pickerVisible") is False,
    "accepted_path_survives_reload": draft_reload.get("value") == "./subdir/child.txt",
    "spacey_directory_option": space_option.get("value") == "./my" and texts(space_option) == ["my dir/"],
    "spacey_directory_drills": space_children.get("value") == "./my dir/" and space_children.get("pickerVisible") is True and space_children.get("header") == "/home/tester/path-completion-fixture/my dir" and texts(space_children) == ["space-child.txt"],
    "index_zero_slash_is_command_picker": index_zero.get("value") == "/" and index_zero.get("pickerVisible") is True and index_zero.get("ariaLabel") == "Available commands" and not index_zero.get("pathOptions"),
    "mid_message_slash_is_absolute_picker": absolute.get("value") == " /e" and absolute.get("pickerVisible") is True and absolute.get("ariaLabel") == "Available paths" and absolute.get("header") == "/" and "etc/" in texts(absolute),
    "url_never_opens_path_picker": url.get("value") == "https://example.com" and not url.get("pickerVisible") and not url.get("pathOptions"),
    "escape_closes_path_picker": before_escape.get("pickerVisible") is True and after_escape.get("pickerVisible") is False,
    "next_input_reopens_path_picker": reopened.get("value") == "./p" and reopened.get("pickerVisible") is True and reopened.get("ariaLabel") == "Available paths",
    "caret_click_leaves_token_and_closes": before_caret.get("pickerVisible") is True and after_caret.get("pickerVisible") is False and isinstance(after_caret.get("caret"), int) and after_caret["caret"] <= 1,
    "mention_opens_picker_at_cwd": mention_open.get("value") == "@" and mention_open.get("pickerVisible") is True and mention_open.get("ariaLabel") == "Available paths" and mention_open.get("header") == "/home/tester/path-completion-fixture" and {"subdir/", "plain.txt"}.issubset(set(texts(mention_open))),
    "mention_fragment_narrows": mention_fragment.get("value") == "@pln" and texts(mention_fragment) == ["plain.txt"],
    "mention_accept_keeps_marker": mention_accepted.get("value") == "@plain.txt" and mention_accepted.get("pickerVisible") is False,
    "mention_email_never_triggers": mention_email.get("value") == "foo@bar" and not mention_email.get("pickerVisible") and not mention_email.get("pathOptions"),
    "mention_no_match_does_not_swallow_enter": mention_nomatch.get("pickerVisible") is True and str(mention_enter.get("value", "")).endswith("\n"),
    "mention_message_renders_a_link": mention_link.get("text") == "@plain.txt" and mention_link.get("kind") == "text" and str(mention_link.get("path", "")).endswith("/plain.txt") and not mention_link.get("pendingLink"),
    "mention_link_opens_the_file": mention_viewer.get("viewerDisplay") not in {None, "none"} and str(mention_viewer.get("path", "")).endswith("plain.txt"),
}
print(json.dumps({"checks": checks, "observed": {"dot_slash": first, "fragment": fragment, "subdir_active": subdir_active, "subdir": subdir, "file_accepted": file_accepted, "draft_reload": draft_reload, "space_option": space_option, "space_children": space_children, "index_zero": index_zero, "absolute": absolute, "url": url, "before_escape": before_escape, "after_escape": after_escape, "reopened": reopened, "before_caret": before_caret, "after_caret": after_caret, "mention_open": mention_open, "mention_fragment": mention_fragment, "mention_accepted": mention_accepted, "mention_email": mention_email, "mention_nomatch": mention_nomatch, "mention_enter": mention_enter, "mention_link": mention_link, "mention_viewer": mention_viewer}, "pass": all(checks.values())}, indent=2))
PY

python3 - "$artifacts" "$target_commit" <<'PY' > "$artifacts/report.json"
import json, sys
from pathlib import Path
artifacts = Path(sys.argv[1])
api = json.loads((artifacts / "api-checks.json").read_text(encoding="utf-8"))
ui = json.loads((artifacts / "ui-checks.json").read_text(encoding="utf-8"))
print(json.dumps({"commit": sys.argv[2], "api": api, "ui": ui, "pass": api["pass"] and ui["pass"]}, indent=2))
PY

cat "$artifacts/api-checks.json"
cat "$artifacts/ui-checks.json"
if ! python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$artifacts/report.json"; then
  echo "FAIL: path-completion verification failed; artifacts=$artifacts" >&2
  exit 1
fi
printf 'PASS: composer path completion verified at %s; artifacts=%s screenshots=%s\n' "$short_commit" "$artifacts" "$artifacts/ui-01-dot-slash.png,$artifacts/ui-05-subdir-children.png,$artifacts/ui-09-spacey-children.png,$artifacts/ui-11-mid-message-absolute.png,$artifacts/ui-17-after-caret-click.png,$artifacts/ui-18-mention-open.png,$artifacts/ui-20-mention-accepted.png,$artifacts/ui-24-mention-link.png,$artifacts/ui-25-mention-viewer.png"
