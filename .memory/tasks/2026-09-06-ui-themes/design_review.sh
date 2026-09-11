#!/usr/bin/env bash
# Docker-isolated screenshot harness for the full Codoxear theme review matrix.
#
# The target application is always built from a Git archive.  Its Pi config and
# synthetic content fixture live in a throwaway HOME, while the host browser is
# restricted to an explicitly non-live loopback port.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: .memory/tasks/2026-09-06-ui-themes/design_review.sh [commit]

Writes 78 viewport screenshots and report.json to a newly-created
/tmp/codoxear-design-review.XXXXXX directory.

Environment:
  CODOXEAR_DESIGN_PORT       Loopback port (default: 19645; never 8743)
  CODOXEAR_DESIGN_IMAGE      Image tag (default: codoxear-design-review:<commit>)
  CODOXEAR_DESIGN_PASSWORD   Login password (default: docker-design-password)
USAGE
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
target_ref="${1:-HEAD}"
target_commit="$(git -C "$repo_root" rev-parse --verify "${target_ref}^{commit}")" || {
  echo "unable to resolve commit: $target_ref" >&2
  exit 2
}
short_commit="$(git -C "$repo_root" rev-parse --short=12 "$target_commit")"
port="${CODOXEAR_DESIGN_PORT:-19645}"
password="${CODOXEAR_DESIGN_PASSWORD:-docker-design-password}"
image="${CODOXEAR_DESIGN_IMAGE:-codoxear-design-review:${short_commit}}"
container="codoxear-design-review-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
fixture_generator="$repo_root/.memory/tasks/2026-09-06-ui-themes/design_fixture.py"
root="$(mktemp -d /tmp/codoxear-design-review-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-design-review.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-design-review-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_DESIGN_PORT must be a TCP port, got: $port" >&2
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
if [[ ! -r "$fixture_generator" ]]; then
  echo "fixture generator is unavailable: $fixture_generator" >&2
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
  "${docker[@]}" logs "$container" > "$artifacts/container.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\\n" "$HOME"
    printf "APP_DIR=%s\\n" "$(python3 -c "from codoxear.util import default_app_dir; print(default_app_dir())")"
    find "$HOME/.pi/agent/sessions" -type f -name "*.jsonl" -printf "%p\\n" 2>/dev/null
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -printf "%p\\n" 2>/dev/null
  ' > "$artifacts/isolation-and-session.txt" 2>&1 || true
}

cleanup() {
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  capture_container_diagnostics
  echo "artifacts retained at $artifacts" >&2
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

wait_for_server() {
  local code
  for _ in $(seq 1 120); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
    [[ "$code" == "401" ]] && return 0
    sleep 1
  done
  return 1
}

wait_for_pi_session() {
  local cookie="$root/cookies.txt"
  curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
    --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"
  for _ in $(seq 1 120); do
    if curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions.json" 2>/dev/null \
      && python3 - "$artifacts/sessions.json" <<'PY'
import json
import sys
sessions = json.load(open(sys.argv[1], encoding="utf-8")).get("sessions", [])
raise SystemExit(0 if any(isinstance(row, dict) and row.get("agent_backend") == "pi" for row in sessions) else 1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

# A single capture stores the exact DOM theme/mode alongside its viewport.  The
# final JSON report is built from these raw agent-browser responses, preserving
# the probe from the instant the PNG was written.
capture() {
  local name="$1" viewport="$2"
  browser eval '(() => ({
    theme: document.documentElement.dataset.theme || null,
    dataMode: document.documentElement.dataset.mode || null,
    viewport: { width: innerWidth, height: innerHeight },
    selectedTitle: document.querySelector("#threadTitle")?.textContent || null,
    focusedId: document.activeElement?.id || null,
    focusedSelector: document.activeElement?.className || null,
    focusVisible: Boolean(document.activeElement && document.activeElement.matches && document.activeElement.matches(":focus-visible")),
    filePickerVisible: (() => { const n = document.querySelector("#filePickerMenu"); return Boolean(n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0); })(),
    filePickerOptionCount: document.querySelectorAll("#filePickerMenu [role=option]").length,
    fileViewerVisible: (() => { const n = document.querySelector("#fileViewer"); return Boolean(n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0); })(),
    editVisible: (() => { const n = document.querySelector("#editViewer"); return Boolean(n && (n.open || (getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0))); })(),
    queueViewerVisible: (() => { const n = document.querySelector("#queueViewer"); return Boolean(n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0); })(),
    unattendedVisible: (() => { const n = document.querySelector("#unattendedMenu"); return Boolean(n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0); })(),
    modelPickerVisible: (() => { const n = document.querySelector("#modelPicker"); return Boolean(n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0); })(),
    modelPickerOptionCount: document.querySelectorAll("#modelPicker [role=option]").length
  }))()' --json > "$artifacts/${name}.probe.json" 2>&1 || fail "theme probe failed for $name"
  browser screenshot "$artifacts/${name}.png" > /dev/null 2>&1 || fail "screenshot failed for $name"
  printf '%s\t%s\n' "$name" "$viewport" >> "$artifacts/captures.tsv"
}

select_fixture() {
  # The fixture's metadata contains claude-fable-5-1. This intentionally
  # selects the content card rather than relying on sidebar position.
  browser eval '(() => {
    const card = [...document.querySelectorAll("#sessions .session")].find((el) => /fable/i.test(el.innerText || ""));
    if (!card) throw new Error("no sidebar card contains fable metadata");
    card.click();
    return { text: card.innerText, selected: true };
  })()' --json > /dev/null 2>&1 || fail "fixture sidebar card (fable metadata) was not clickable"
  browser wait 900 > /dev/null 2>&1
  browser eval '(() => { const chat = document.querySelector("#chat"); if (chat) chat.scrollTop = 0; })()' > /dev/null 2>&1 || fail "could not scroll transcript to its top"
  browser wait 150 > /dev/null 2>&1
}

switch_variant() {
  local family="$1" mode="$2"
  browser eval "(() => { localStorage.setItem('codoxear.ui.theme.family', '${family}'); localStorage.setItem('codoxear.ui.theme.mode', '${mode}'); })()" > /dev/null 2>&1 || fail "could not write ${family}/${mode} theme preference"
  browser reload > /dev/null 2>&1 || fail "reload failed for ${family}/${mode}"
  browser wait 2500 > /dev/null 2>&1
  select_fixture
}

close_surface() {
  local selector="$1"
  browser click "$selector" > /dev/null 2>&1 || fail "could not close surface with $selector"
  browser wait 250 > /dev/null 2>&1
}

capture_variant() {
  local family="$1" mode="$2" prefix="${1}-${2}"
  switch_variant "$family" "$mode"

  browser set viewport 1280 860 > /dev/null 2>&1 || fail "could not set desktop viewport"
  browser eval '(() => { const chat = document.querySelector("#chat"); if (chat) chat.scrollTop = 0; })()' > /dev/null 2>&1
  browser wait 150 > /dev/null 2>&1
  capture "${prefix}-app" desktop

  browser click '#settingsBtnSide' > /dev/null 2>&1 || fail "could not open Settings"
  browser wait 300 > /dev/null 2>&1
  capture "${prefix}-settings" desktop
  close_surface '#settingsCloseBtn'

  browser click '#newBtn' > /dev/null 2>&1 || fail "could not open New session"
  browser wait 300 > /dev/null 2>&1
  capture "${prefix}-newsession" desktop
  close_surface '#newSessionCloseBtn'

  browser click '#helpBtnSide' > /dev/null 2>&1 || fail "could not open Help"
  browser wait 300 > /dev/null 2>&1
  capture "${prefix}-help" desktop
  close_surface '#helpCloseBtn'

  browser click '#diagBtn' > /dev/null 2>&1 || fail "could not open Details"
  browser wait 500 > /dev/null 2>&1
  capture "${prefix}-diag" desktop
  close_surface '#diagCloseBtn'

  browser hover '#sessions .session' > /dev/null 2>&1 || browser mouse move 120 120 > /dev/null 2>&1 || fail "could not hover first session card"
  browser wait 250 > /dev/null 2>&1
  capture "${prefix}-hover-card" desktop
  browser hover '#fileBtn' > /dev/null 2>&1 || browser mouse move 1180 28 > /dev/null 2>&1 || fail "could not hover topbar file button"
  browser wait 250 > /dev/null 2>&1
  capture "${prefix}-hover-icon" desktop
  browser click '#msg' > /dev/null 2>&1 || fail "could not focus composer for focus-ring capture"
  browser press Tab > /dev/null 2>&1 || fail "could not move focus to a composer control"
  browser wait 150 > /dev/null 2>&1
  capture "${prefix}-focus-ring" desktop
  browser click '#queueBtn' > /dev/null 2>&1 || fail "could not open queue viewer"
  browser wait 500 > /dev/null 2>&1
  capture "${prefix}-queueviewer" desktop
  close_surface '#queueCloseBtn'
  browser fill '#msg' '/' > /dev/null 2>&1 || fail "could not open slash completion"
  browser wait 500 > /dev/null 2>&1
  capture "${prefix}-slashmenu" desktop
  browser fill '#msg' '' > /dev/null 2>&1 || fail "could not clear slash completion"
  browser click '#unattendedBtn' > /dev/null 2>&1 || fail "could not open unattended menu"
  browser wait 500 > /dev/null 2>&1
  capture "${prefix}-unattended" desktop
  browser click '#unattendedBtn' > /dev/null 2>&1 || fail "could not close unattended menu"
  browser wait 200 > /dev/null 2>&1
  browser fill '#msg' '/model ' > /dev/null 2>&1 || fail "could not probe Pi model picker"
  browser wait 500 > /dev/null 2>&1
  model_picker_visible="$(browser eval '(() => { const n = document.querySelector("#modelPicker"); return Boolean(n && getComputedStyle(n).display !== "none" && n.getBoundingClientRect().width > 0 && n.getBoundingClientRect().height > 0); })()' --json 2>/dev/null || true)"
  if python3 - "$model_picker_visible" <<'PY'
import json, sys
try:
    value = json.loads(sys.argv[1]).get("data", {}).get("result")
except Exception:
    value = False
raise SystemExit(0 if value is True else 1)
PY
  then
    capture "${prefix}-modelpicker" desktop
  else
    printf '%s\n' "{\"surface\":\"${prefix}-modelpicker\",\"reason\":\"Pi model picker was not reachable offline\",\"domProbe\":${model_picker_visible:-null}}" >> "$artifacts/file-surface-skips.jsonl"
  fi
  browser fill '#msg' '' > /dev/null 2>&1 || fail "could not clear model picker probe"

  browser click '#chatSearchBtn' > /dev/null 2>&1 || fail "could not open conversation search"
  browser fill '#chatSearchInput' 'token' > /dev/null 2>&1 || fail "could not search fixture transcript"
  browser wait 600 > /dev/null 2>&1
  capture "${prefix}-search" desktop
  close_surface '#chatSearchCloseBtn'

  browser click '#fileBtn' > /dev/null 2>&1 || fail "could not open file viewer"
  browser click '#filePickerInput' > /dev/null 2>&1 || fail "could not focus file picker"
  browser wait 500 > /dev/null 2>&1
  capture "${prefix}-filepicker" desktop
  browser fill '#filePickerInput' 'pyproject.toml' > /dev/null 2>&1 || fail "could not search for pyproject.toml"
  browser wait 1000 > /dev/null 2>&1
  browser eval '(() => {
    const options = [...document.querySelectorAll("#filePickerMenu [role=option]")];
    const target = options.find((el) => /pyproject\.toml/i.test((el.innerText || "")));
    if (!target) return { found: false, menuVisible: Boolean(document.querySelector("#filePickerMenu") && getComputedStyle(document.querySelector("#filePickerMenu")).display !== "none"), options: options.map((el) => (el.innerText || "").trim()) };
    target.click();
    return { found: true, path: (target.innerText || "").trim() };
  })()' --json > "$artifacts/${prefix}-file-selection.json" 2>&1 || fail "file picker selection probe failed"
  if python3 - "$artifacts/${prefix}-file-selection.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
result = payload.get("data", {}).get("result", {}) if isinstance(payload, dict) else {}
raise SystemExit(0 if result.get("found") else 1)
PY
  then
    browser wait 1400 > /dev/null 2>&1
    capture "${prefix}-fileviewer" desktop
  else
    python3 - "$artifacts/${prefix}-file-selection.json" "$artifacts/file-surface-skips.jsonl" "${prefix}-fileviewer" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
result = payload.get("data", {}).get("result", {}) if isinstance(payload, dict) else {}
with open(sys.argv[2], "a", encoding="utf-8") as out:
    out.write(json.dumps({"surface": sys.argv[3], "reason": "pyproject.toml candidate was absent from the rendered picker", "dom": result}) + "\\n")
PY
  fi
  close_surface '#fileCloseBtn'

  browser click '#threadTitle' > /dev/null 2>&1 || fail "could not open Edit conversation dialog"
  browser wait 400 > /dev/null 2>&1
  capture "${prefix}-edit" desktop
  close_surface '#editCloseBtn'

  browser click '#ctxChip' > /dev/null 2>&1 || fail "fixture token context chip was not clickable"
  browser wait 100 > /dev/null 2>&1
  capture "${prefix}-toast" desktop
  # Let the transient toast expire before the independent composer surface.
  browser wait 2400 > /dev/null 2>&1

  browser focus '#msg' > /dev/null 2>&1 || fail "could not focus composer"
  browser type '#msg' 'line one' > /dev/null 2>&1 || fail "could not type first composer line"
  browser press 'Shift+Enter' > /dev/null 2>&1 || fail "could not insert composer newline"
  browser type '#msg' 'line two' > /dev/null 2>&1 || fail "could not type second composer line"
  browser wait 150 > /dev/null 2>&1
  capture "${prefix}-composer" desktop
  # Drafts are intentionally persisted by the product. Clear this review-only
  # input before reloading, otherwise the next variant would append to the
  # previous multiline probe and contaminate its app/mobile surfaces.
  browser eval '(() => { const t = document.querySelector("#msg"); if (!t) throw new Error("composer missing"); t.value = ""; t.dispatchEvent(new Event("input", { bubbles: true })); })()' > /dev/null 2>&1 || fail "could not clear review composer draft"

  # Reload before the mobile pair so its app surface represents a clean
  # transcript/composer state rather than the desktop multiline probe above.
  browser reload > /dev/null 2>&1 || fail "reload failed before mobile captures"
  browser wait 2500 > /dev/null 2>&1
  select_fixture
  browser set viewport 390 844 > /dev/null 2>&1 || fail "could not set mobile viewport"
  browser wait 300 > /dev/null 2>&1
  browser eval '(() => { const chat = document.querySelector("#chat"); if (chat) chat.scrollTop = 0; })()' > /dev/null 2>&1
  capture "${prefix}-app-mobile" mobile
  browser click '#toggleSidebarBtn' > /dev/null 2>&1 || fail "could not open mobile sidebar for Settings"
  browser wait 150 > /dev/null 2>&1
  browser click '#settingsBtnSide' > /dev/null 2>&1 || fail "could not open mobile Settings"
  browser wait 300 > /dev/null 2>&1
  capture "${prefix}-settings-mobile" mobile
  close_surface '#settingsCloseBtn'
  browser set viewport 1280 860 > /dev/null 2>&1 || fail "could not restore desktop viewport"
}

copy_pi_config
fixture_path="$(python3 "$fixture_generator" "$home_dir/.pi/agent/sessions/--workspace--")" || fail "fixture generation failed"
[[ -f "$fixture_path" ]] || fail "fixture generator did not create a session log"
printf '%s\n' "$fixture_path" > "$artifacts/fixture-path.txt"
fixture_session_id="$(python3 - "$fixture_path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    row = json.loads(stream.readline())
print(row["id"])
PY
)" || fail "could not read fixture session id"
mkdir -p "$home_dir/.local/share/codoxear"
python3 - "$home_dir/.local/share/codoxear/session_queues.json" "$fixture_session_id" <<'PY'
import json, sys, time
path, sid = sys.argv[1:]
items = [
    {"id": "design-review-queue-1", "text": "Review the queued theme token extraction notes.", "created_ts": time.time() - 2, "orphan_recovery": True},
    {"id": "design-review-queue-2", "text": "Compare the dark palette against the paper baseline.", "created_ts": time.time() - 1, "orphan_recovery": True},
]
with open(path, "w", encoding="utf-8") as stream:
    json.dump({sid: items}, stream)
    stream.write("\n")
PY
printf '%s\n' "$fixture_session_id" > "$artifacts/fixture-session-id.txt"
: > "$artifacts/file-surface-skips.jsonl"

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
  "$image" \
  bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    fixture_log="$(find "$HOME/.pi/agent/sessions/--workspace--" -type f -name "*.jsonl" -print -quit)"
    test -n "$fixture_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$fixture_log" >"$HOME/sessiond.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"; then
        break
      fi
      sleep 1
    done
    test -n "${meta:-}" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"
    # The web-facing session identifier is the broker/socket identifier, while the Pi log native
    # thread id is the fixture UUID. Re-key the pre-seeded queue map to
    # the discovered web id before the server loads persistent state.
    python3 - "$HOME/.local/share/codoxear/session_queues.json" "$meta" <<PY
import json, sys
from pathlib import Path
queue_path, meta_path = sys.argv[1:]
with open(meta_path, encoding="utf-8") as stream:
    metadata = json.load(stream)
web_id = Path(meta_path).stem or metadata.get("session_id")
with open(queue_path, encoding="utf-8") as stream:
    queues = json.load(stream)
if web_id and web_id not in queues and queues:
    old_id, items = next(iter(queues.items()))
    queues[web_id] = items
    if old_id != web_id:
        queues.pop(old_id, None)
with open(queue_path, "w", encoding="utf-8") as stream:
    json.dump(queues, stream)
    stream.write("\n")
PY
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "server readiness check failed"
wait_for_pi_session || fail "Pi fixture session never appeared"

browser set viewport 1280 860 > /dev/null 2>&1 || true
browser set media light > /dev/null 2>&1 || true
browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open app"
browser errors --clear --json > /dev/null 2>&1 || true
browser console --clear --json > /dev/null 2>&1 || true
browser fill '#pw' "$password" > /dev/null 2>&1 || fail "login password field was not available"
browser click '#loginBtn' > /dev/null 2>&1 || fail "login button was not available"
browser wait 3000 > /dev/null 2>&1
select_fixture
: > "$artifacts/captures.tsv"

for family in paper clay slate; do
  for mode in light dark; do
    capture_variant "$family" "$mode"
  done
done

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true
capture_container_diagnostics

python3 - "$artifacts" "$target_commit" <<'PY'
import json
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])
commit = sys.argv[2]
rows = []
for raw in (artifacts / "captures.tsv").read_text(encoding="utf-8").splitlines():
    name, viewport = raw.split("\t", 1)
    payload = json.loads((artifacts / f"{name}.probe.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise SystemExit(f"unsuccessful probe for {name}: {payload!r}")
    probe = payload.get("data", {}).get("result")
    if not isinstance(probe, dict):
        raise SystemExit(f"missing probe result for {name}: {payload!r}")
    path = artifacts / f"{name}.png"
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing screenshot: {path}")
    rows.append({"path": str(path), "viewport": viewport, "probe": probe})
try:
    skipped = [json.loads(line) for line in (artifacts / "file-surface-skips.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
except FileNotFoundError:
    skipped = []
expected = 6 * 18 + 6 * 2 - len(skipped)
if len(rows) != expected:
    raise SystemExit(f"expected {expected} captures, found {len(rows)}")
mismatches = [
    row for row in rows
    if (lambda parts: row["probe"].get("theme") != parts[0] or row["probe"].get("dataMode") != parts[1])(
        Path(row["path"]).stem.removesuffix("-mobile").split("-", 2)
    )
]
report = {
    "pass": not mismatches,
    "commit": commit,
    "artifactDir": str(artifacts),
    "captureCount": len(rows),
    "captures": rows,
    "mismatches": mismatches,
    "surfaceSkips": skipped,
}
(artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pass": report["pass"], "captureCount": len(rows), "mismatches": len(mismatches), "surfaceSkips": len(skipped)}, indent=2))
if mismatches:
    raise SystemExit(1)
PY

printf 'PASS: commit=%s artifacts=%s\n' "$target_commit" "$artifacts"
