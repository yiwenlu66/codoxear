#!/usr/bin/env bash
# Docker-isolated behavioral verifier for the new-session provider/model
# combobox: opening the list clears a committed pair and lists every configured
# model grouped by provider, the first edit of a committed pair replaces it, and
# a browse-only open that ends without a selection restores the pair — including
# before Start, so browsing can never silently launch the default model.
#
# The application is built from a Git archive of the requested commit (plus a
# regenerated browser bundle) and runs under a container-only HOME. The host
# browser may connect only to a loopback port other than 8743; no host Codoxear
# runtime or deployment state is mounted.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_new_session_model_picker_verify.sh [commit]

Writes report.json, browser probes, screenshots, and diagnostics to a new
/tmp/codoxear-model-combobox-verify.* directory.

Environment:
  CODOXEAR_MODEL_COMBOBOX_PORT       Loopback port (default: 19671; never 8743)
  CODOXEAR_MODEL_COMBOBOX_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_MODEL_COMBOBOX_PASSWORD   Login password (default: docker-combobox-password)
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
port="${CODOXEAR_MODEL_COMBOBOX_PORT:-19671}"
password="${CODOXEAR_MODEL_COMBOBOX_PASSWORD:-docker-combobox-password}"
image="${CODOXEAR_MODEL_COMBOBOX_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-model-combobox-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-model-combobox-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-model-combobox-verify.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-model-combobox-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then echo "refusing live Codoxear port 8743" >&2; exit 2; fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || ((port > 65535)); then echo "invalid loopback port: $port" >&2; exit 2; fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then echo "refusing host app-dir override" >&2; exit 2; fi
if [[ ! -r "$source_pi_agent/models.json" ]]; then
  echo "Pi model config is missing under $source_pi_agent" >&2
  exit 2
fi
command -v agent-browser >/dev/null || { echo "agent-browser is required" >&2; exit 2; }

browser() { AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"; }
capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\n" "$HOME"
    python3 -c "from codoxear.util import default_app_dir; print(default_app_dir())"
    cat "$HOME"/.local/share/codoxear/session_launches.jsonl 2>/dev/null || true
  ' > "$artifacts/launch-records.txt" 2>&1 || true
}
cleanup() {
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  /bin/rm -rf "$root"
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; capture_container_diagnostics; exit 1; }
copy_pi_config() {
  umask 077
  mkdir -p "$home_dir/.pi/agent" "$home_dir/.local/share"
  local name
  for name in models.json models-store.json auth.json settings.json trust.json anthropic-context-management.json; do
    [[ -r "$source_pi_agent/$name" ]] && install -m 600 "$source_pi_agent/$name" "$home_dir/.pi/agent/$name"
  done
}
wait_for_server() {
  local code
  for _ in $(seq 1 120); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
    [[ "$code" == 401 ]] && return 0
    sleep 1
  done
  return 1
}

copy_pi_config
build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"
# The browser loads dist/app.bundle.js; rebuild it from this commit's sources in
# the immutable archive context so the verifier exercises committed source, not
# whatever bundle happens to sit in the checkout.
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"
"${docker[@]}" build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" > "$artifacts/docker-build.log" 2>&1 || fail "Docker build failed"

"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach --name "$container" --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester --env CODEX_WEB_PASSWORD="$password" --env CODEX_WEB_HOST=0.0.0.0 --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DEFAULT_AGENT_BACKEND=pi --env PYTHONPATH=/workspace --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$home_dir,dst=/home/tester" --workdir /workspace "$image" bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    # Two idle sessions in different directories seed recent_cwds, so the cwd
    # picker has a history to browse and a selected session to prefill from.
    for pair in docker-combobox-ws:/workspace docker-combobox-tmp:/tmp; do
      sid="${pair%%:*}"; dir="${pair#*:}"
      pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files --session-id "$sid" -p "Initialize isolated combobox verification session." >"$HOME/pi-bootstrap-$sid.log" 2>&1 || true
      session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*${sid}.jsonl" -print -quit)"
      test -n "$session_log"
      CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 codoxear-broker --cwd "$dir" -- --offline --no-extensions --no-skills --no-prompt-templates --no-context-files --session "$session_log" >"$HOME/sessiond-$sid.log" 2>&1 &
    done
    for _ in $(seq 1 120); do
      n=0
      for meta in "$HOME"/.local/share/codoxear/socks/*.json; do test -f "$meta" && grep -q session_id "$meta" && n=$((n + 1)); done
      test "$n" -ge 2 && break
      sleep 1
    done
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"
wait_for_server || fail "server readiness check failed"

# Configured Pi pairs and the seeded session ids, straight from the API.
curl -fsS -c "$root/cookies.txt" -H 'Content-Type: application/json' \
  --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"
wait_for_seeded_sessions() {
  local have
  for _ in $(seq 1 60); do
    curl -sS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions.json" 2>/dev/null || true
    have="$(python3 - "$artifacts/sessions.json" <<'PY' 2>/dev/null || true
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
cwds = {str(s.get("cwd") or "") for s in payload.get("sessions", []) if isinstance(s, dict)}
print(1 if {"/workspace", "/tmp"} <= cwds else 0)
PY
)"
    [[ "$have" == 1 ]] && return 0
    sleep 1
  done
  return 1
}
wait_for_seeded_sessions || fail "seeded sessions never appeared"
python3 - "$artifacts/sessions.json" > "$artifacts/expected-counts.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
defaults = (payload.get("new_session_defaults") or {}).get("backends") or {}

def counts(backend):
    entry = defaults.get(backend) or {}
    provider_models = entry.get("provider_models") or {}
    pairs = [{"model": model, "provider": provider} for provider, models in provider_models.items() if isinstance(models, list) for model in models]
    return {"pairs": pairs}

sessions = [s for s in payload.get("sessions", []) if isinstance(s, dict)]
print(json.dumps({
    "pi": counts("pi"),
    "codex": counts("codex"),
    "recent_cwds": [str(c) for c in (payload.get("recent_cwds") or [])],
    "session_cwds": {str(s.get("cwd") or ""): str(s.get("session_id") or "") for s in sessions},
}, indent=2))
PY

PROBE='(() => {
  const q = (s) => document.querySelector(s);
  const input = q("#newSessionModelInput"), menu = q("#newSessionModelMenu"), viewer = q("#newSessionViewer");
  const cwdInput = q("#newSessionCwdInput"), cwdMenu = q("#newSessionCwdMenu");
  const options = menu ? [...menu.querySelectorAll("[role=option]")] : [];
  const activeId = input ? input.getAttribute("aria-activedescendant") : null;
  const activeEl = activeId ? document.getElementById(activeId) : null;
  const cwdOptions = cwdMenu ? [...cwdMenu.querySelectorAll("[role=option]")] : [];
  const cwdActiveId = cwdInput ? cwdInput.getAttribute("aria-activedescendant") : null;
  const cwdActiveEl = cwdActiveId ? document.getElementById(cwdActiveId) : null;
  return {
    dialogOpen: Boolean(viewer && getComputedStyle(viewer).display !== "none"),
    value: input ? input.value : null,
    expanded: input ? input.getAttribute("aria-expanded") : null,
    menuDisplay: menu ? getComputedStyle(menu).display : null,
    optionCount: options.length,
    options: options.map((n) => ({ id: n.id, label: n.getAttribute("aria-label") })),
    groups: menu ? [...menu.querySelectorAll(".modelMenuGroup")].map((n) => n.textContent) : [],
    groupHeaderCount: menu ? menu.querySelectorAll(".modelMenuGroup").length : 0,
    active: activeEl ? activeEl.getAttribute("aria-label") : null,
    cwdValue: cwdInput ? cwdInput.value : null,
    cwdMenuDisplay: cwdMenu ? getComputedStyle(cwdMenu).display : null,
    cwdOptionCount: cwdOptions.length,
    cwdOptions: cwdOptions.map((n) => n.getAttribute("title")),
    cwdActive: cwdActiveEl ? cwdActiveEl.getAttribute("title") : null,
    reasoning: (q("#newSessionReasoningBtn") || {}).textContent || null,
    inlineMaxHeight: menu ? menu.style.maxHeight : null,
    renderedHeight: menu ? Math.round(menu.getBoundingClientRect().height) : null,
    clipping: menu ? getComputedStyle(menu).overflowY : null
  };
})()'
probe() { browser eval "$PROBE" --json > "$artifacts/$1.json" 2>&1 || fail "probe $1 failed"; }
evalv() {
  browser eval "$1" --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["result"])' 2>/dev/null || true
}
shot() { browser screenshot "$artifacts/$1.png" > /dev/null 2>&1 || fail "screenshot $1 failed"; }
settle() { browser wait "${1:-450}" > /dev/null 2>&1; }
wait_for_dialog() {
  local seen
  for _ in $(seq 1 40); do
    seen="$(evalv '(() => { const i=document.getElementById("newSessionModelInput"); const v=document.getElementById("newSessionViewer"); return Boolean(i && v && getComputedStyle(v).display !== "none" && String(i.value||"").length > 0); })()')"
    [[ "$seen" == True ]] && return 0
    sleep 0.15
  done
  return 1
}
wait_for_menu() {
  local seen
  for _ in $(seq 1 40); do
    seen="$(evalv '(() => { const m=document.getElementById("newSessionModelMenu"); return Boolean(m && getComputedStyle(m).display === "block"); })()')"
    [[ "$seen" == True ]] && return 0
    sleep 0.15
  done
  return 1
}
wait_for_cwd_menu() {
  local seen
  for _ in $(seq 1 40); do
    seen="$(evalv '(() => { const m=document.getElementById("newSessionCwdMenu"); return Boolean(m && getComputedStyle(m).display === "block"); })()')"
    [[ "$seen" == True ]] && return 0
    sleep 0.15
  done
  return 1
}
wait_for_dialog_cwd() {
  local seen
  for _ in $(seq 1 40); do
    seen="$(evalv '(() => { const i=document.getElementById("newSessionCwdInput"); const v=document.getElementById("newSessionViewer"); return Boolean(i && v && getComputedStyle(v).display !== "none" && String(i.value||"").trim().length > 0); })()')"
    [[ "$seen" == True ]] && return 0
    sleep 0.15
  done
  return 1
}
select_session() { browser click "#sessions .session[data-session-id=\"$1\"]" >/dev/null 2>&1 || fail "session selection failed"; settle 1200; }
wait_for_start_enabled() {
  for _ in $(seq 1 60); do
    [[ "$(evalv 'document.getElementById("newSessionStartBtn").disabled === false')" == True ]] && return 0
    sleep 0.5
  done
  return 1
}
close_dialog_if_open() {
  local open
  open="$(evalv '(() => { const v=document.getElementById("newSessionViewer"); return Boolean(v && getComputedStyle(v).display !== "none"); })()')"
  [[ "$open" == True ]] && { browser click '#newSessionCloseBtn' >/dev/null 2>&1 || fail "could not close the dialog"; settle; }
  return 0
}
launch_record_count() { "${docker[@]}" exec "$container" sh -lc 'wc -l < "$HOME/.local/share/codoxear/session_launches.jsonl" 2>/dev/null || echo 0' | tr -d '[:space:]'; }
open_dialog() { browser click '#newBtn' >/dev/null 2>&1 || fail "could not open the new-session dialog"; wait_for_dialog || fail "new-session dialog never showed a committed model pair"; settle; }

browser set viewport 1280 1000 >/dev/null 2>&1 || true
browser set media light >/dev/null 2>&1 || true
browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser open failed"
browser errors --clear --json > "$artifacts/browser-errors-before.json" 2>&1 || true
browser console --clear --json > "$artifacts/browser-console-before.json" 2>&1 || true
browser fill '#pw' "$password" >/dev/null 2>&1 || fail "login password unavailable"
browser click '#loginBtn' >/dev/null 2>&1 || fail "login button unavailable"
settle 3500

# 1. The dialog opens on a committed pair, menu closed.
open_dialog
probe 01-initial

# 2. Clicking the field clears the pair and lists every configured model,
#    recents first and grouped per provider, with the committed pair highlighted.
browser click '#newSessionModelInput' >/dev/null 2>&1 || fail "could not click the model field"
wait_for_menu || fail "model menu did not open on click"
settle 600
probe 02-browse
shot 01-browse

# 3. Dismissing a browse-only open restores the committed pair.
browser press Escape >/dev/null 2>&1 || fail "Escape failed"
settle 400
probe 03-dismiss-restored

# 4. The first edit of a committed pair replaces it instead of appending, and
#    a query result list stays ranked and capped.
browser type '#newSessionModelInput' 'c' >/dev/null 2>&1 || fail "could not type over the committed pair"
settle 400
probe 04-replace-on-type

# 5. A typed query is the user's content: dismissing keeps it.
browser press Escape >/dev/null 2>&1 || fail "Escape failed after typing"
settle 300
probe 05-query-kept

# 6. A fresh dialog, then browse and pick a different option.
browser click '#newSessionCloseBtn' >/dev/null 2>&1 || fail "could not close the dialog"
settle
open_dialog
probe 06-reopened
browser click '#newSessionModelInput' >/dev/null 2>&1 || fail "could not reopen the model list"
wait_for_menu || fail "model menu did not reopen"
settle 500
probe 07-browse-again
browser click '#newSessionModelOption-1' >/dev/null 2>&1 || fail "could not select the second model option"
settle 400
probe 08-selected

# 7. Closing the list by clicking outside the field restores the pair.
browser click '#newSessionModelInput' >/dev/null 2>&1 || fail "could not reopen for the outside-click check"
wait_for_menu || fail "model menu did not open before the outside click"
settle 300
browser click '#newSessionCwdInput' >/dev/null 2>&1 || fail "could not click the cwd field"
settle 400
probe 09-outside-click-restored

# 8. A backend switch repopulates a committed pair (and clears browse state).
browser click '#newSessionBackendTabs .agentBackendTab:nth-of-type(2)' >/dev/null 2>&1 || fail "could not switch to the Codex tab"
settle 400
probe 10-codex-committed
browser click '#newSessionModelInput' >/dev/null 2>&1 || fail "could not click the codex model field"
settle 400
probe 11-codex-browse
browser press Escape >/dev/null 2>&1 || fail "Escape failed on the codex list"
settle 300

# 9. Switching back restores the Pi pair, and Start while browse-cleared must
#    launch that pair rather than the launch default.
browser click '#newSessionBackendTabs .agentBackendTab:nth-of-type(1)' >/dev/null 2>&1 || fail "could not switch back to the Pi tab"
settle 400
probe 12-pi-restored
browser click '#newSessionModelInput' >/dev/null 2>&1 || fail "could not clear before Start"
wait_for_menu || fail "model menu did not open before Start"
settle 300
probe 13-before-start
browser fill '#newSessionCwdInput' '/workspace' >/dev/null 2>&1 || fail "could not fill the working directory"
settle 400
browser click '#newSessionStartBtn' >/dev/null 2>&1 || fail "could not click Start"
for _ in $(seq 1 90); do
  [[ "$(launch_record_count)" -ge 1 ]] && break
  sleep 1
done
settle 600
probe 14-after-start

# 10. The cwd picker: opening it on a committed path browses the whole recent
#     history instead of filtering it down to the shown directory.
close_dialog_if_open
curl -fsS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-cwd.json" || fail "could not re-read /api/sessions"
SID_TMP="$(python3 - "$artifacts/sessions-cwd.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for session in payload.get("sessions", []):
    if isinstance(session, dict) and str(session.get("cwd") or "") == "/tmp":
        print(session.get("session_id") or "")
        break
PY
)"
[[ -n "$SID_TMP" ]] || fail "no seeded /tmp session to select"
select_session "$SID_TMP"
browser click '#newBtn' >/dev/null 2>&1 || fail "could not open the new-session dialog from the topbar"
wait_for_dialog_cwd || fail "dialog did not prefill the selected session's path"
settle 400
probe 15-cwd-initial
browser click '#newSessionCwdInput' >/dev/null 2>&1 || fail "could not click the cwd field"
wait_for_cwd_menu || fail "cwd menu did not open on click"
settle 500
probe 16-cwd-browse
shot 02-cwd-browse
browser press Escape >/dev/null 2>&1 || fail "Escape failed on the cwd list"
settle 400
probe 17-cwd-dismiss-restored

# 11. Typing a path still browses the filesystem (the picker only cleared it).
browser fill '#newSessionCwdInput' '/us' >/dev/null 2>&1 || fail "could not type a cwd prefix"
settle 1200
probe 18-cwd-typed-filesystem

# 12. Start while the cwd field is browse-cleared must launch the restored path.
#     The button is activated programmatically so no mousedown closes the list
#     first: this exercises start()'s own restore, not the outside-click path.
browser click '#newSessionCloseBtn' >/dev/null 2>&1 || fail "could not close the dialog"
settle
browser click '#newBtn' >/dev/null 2>&1 || fail "could not reopen the dialog"
wait_for_dialog_cwd || fail "dialog did not prefill the path on reopen"
settle 400
browser click '#newSessionCwdInput' >/dev/null 2>&1 || fail "could not clear the cwd before Start"
wait_for_cwd_menu || fail "cwd menu did not open before Start"
settle 300
probe 19-cwd-before-start
wait_for_start_enabled || fail "Start never became available"
browser eval 'document.getElementById("newSessionStartBtn").click(); true' --json >/dev/null 2>&1 || fail "could not activate Start"
for _ in $(seq 1 90); do
  [[ "$(launch_record_count)" -ge 2 ]] && break
  sleep 1
done
settle 600
probe 20-cwd-after-start

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true
capture_container_diagnostics

python3 - "$artifacts" <<'PY'
import json, sys
from pathlib import Path

artifacts = Path(sys.argv[1])

def load(name):
    payload = json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
    if payload.get("success") is not True:
        raise SystemExit(f"unsuccessful probe {name}: {payload!r}")
    return payload["data"]["result"]

expected = json.loads((artifacts / "expected-counts.json").read_text(encoding="utf-8"))
names = [
    "01-initial", "02-browse", "03-dismiss-restored", "04-replace-on-type",
    "05-query-kept", "06-reopened", "07-browse-again", "08-selected",
    "09-outside-click-restored", "10-codex-committed", "11-codex-browse",
    "12-pi-restored", "13-before-start", "14-after-start",
    "15-cwd-initial", "16-cwd-browse", "17-cwd-dismiss-restored",
    "18-cwd-typed-filesystem", "19-cwd-before-start", "20-cwd-after-start",
]
p = {name: load(name) for name in names}

initial = (p["01-initial"]["value"] or "").strip()
browse = p["02-browse"]
restored = p["03-dismiss-restored"]
replaced = p["04-replace-on-type"]
query_kept = p["05-query-kept"]
browse_again = p["07-browse-again"]
selected = p["08-selected"]
outside = p["09-outside-click-restored"]
codex_committed = p["10-codex-committed"]
codex_browse = p["11-codex-browse"]
pi_restored = p["12-pi-restored"]
before_start = p["13-before-start"]

chosen = (selected["value"] or "").strip()
second_option = browse_again["options"][1]["label"] if len(browse_again["options"]) > 1 else None
pi_pairs = expected["pi"]["pairs"]
configured_labels = {f"{pair['model']} · {pair['provider']}" for pair in pi_pairs}
browse_labels = [item["label"] for item in browse["options"]]

cwd_initial = p["15-cwd-initial"]
cwd_browse = p["16-cwd-browse"]
cwd_restored = p["17-cwd-dismiss-restored"]
cwd_typed = p["18-cwd-typed-filesystem"]
cwd_before_start = p["19-cwd-before-start"]
recent_cwds = {str(cwd) for cwd in (expected.get("recent_cwds") or [])}

# The launch request the app recorded for the browse-cleared Start.
records = []
raw_records = (artifacts / "launch-records.txt").read_text(encoding="utf-8", errors="replace").splitlines()
for line in raw_records:
    line = line.strip()
    if line.startswith("{"):
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
models_by_launch = {}
for record in records:
    launch_id = record.get("launch_id")
    if launch_id: models_by_launch[launch_id] = record
last_record = records[-1] if records else {}
last_cwd_record = next((r for r in reversed(records) if str(r.get("cwd") or r.get("requested_cwd") or "").strip()), {})

def model_of(display):
    return (display or "").split(" · ", 1)[0].strip()

def provider_of(display):
    parts = (display or "").split(" · ", 1)
    return parts[1].strip() if len(parts) > 1 else ""

checks = {
    "dialog_opens_on_a_committed_pair": p["01-initial"]["dialogOpen"] and bool(initial) and p["01-initial"]["menuDisplay"] == "none",
    "click_clears_the_committed_pair": browse["value"] == "",
    "click_opens_the_listbox": browse["menuDisplay"] == "block" and browse["expanded"] == "true",
    "browse_lists_every_configured_model": bool(configured_labels) and configured_labels <= set(browse_labels) and len(set(browse_labels)) == len(browse_labels),
    "browse_is_not_capped_at_twelve": browse["optionCount"] > 12 if len(configured_labels) > 12 else len(configured_labels) <= browse["optionCount"],
    "browse_groups_models_by_provider": browse["groupHeaderCount"] >= 1 and bool(browse["groups"]),
    "browse_highlights_the_committed_pair": browse["active"] == initial,
    "browse_scrolls_within_the_menu": browse["clipping"] == "auto" and bool(browse["inlineMaxHeight"]),
    "dismiss_restores_the_committed_pair": restored["value"] == initial and restored["menuDisplay"] == "none",
    "first_edit_replaces_the_committed_pair": replaced["value"] == "c",
    "typed_query_is_ranked_and_capped": 0 < replaced["optionCount"] <= 12,
    "typed_query_survives_a_dismiss": query_kept["value"] == "c",
    "selection_commits_the_clicked_pair": second_option is not None and chosen == second_option and selected["menuDisplay"] == "none",
    "outside_click_restores_the_committed_pair": outside["value"] == chosen and outside["menuDisplay"] == "none",
    "backend_switch_repopulates_a_committed_pair": bool((codex_committed["value"] or "").strip()) and codex_committed["value"] != chosen,
    "backend_switch_pair_is_cleared_by_a_click": codex_browse["value"] == "",
    "switching_back_restores_the_pi_pair": (pi_restored["value"] or "").strip() == chosen,
    "start_field_was_browse_cleared": before_start["value"] == "",
    "start_launched_the_committed_pair": bool(last_record)
        and last_record.get("model") == model_of(chosen)
        and (last_record.get("model_provider") or last_record.get("provider_choice") or "") == provider_of(chosen),
    "cwd_dialog_prefills_the_selected_session_path": (cwd_initial["cwdValue"] or "").strip() == "/tmp",
    "cwd_click_clears_the_committed_path": cwd_browse["cwdValue"] == "",
    "cwd_click_opens_the_listbox": cwd_browse["cwdMenuDisplay"] == "block",
    "cwd_browse_lists_the_recent_history": cwd_browse["cwdOptionCount"] >= 2
        and set(cwd_browse["cwdOptions"]) <= recent_cwds,
    "cwd_browse_does_not_scrape_the_filesystem_root": "/usr" not in (cwd_browse["cwdOptions"] or []),
    "cwd_browse_highlights_the_committed_path": (cwd_browse["cwdActive"] or "").strip() == "/tmp",
    "cwd_dismiss_restores_the_committed_path": (cwd_restored["cwdValue"] or "").strip() == "/tmp"
        and cwd_restored["cwdMenuDisplay"] == "none",
    "cwd_typing_still_browses_the_filesystem": "/usr" in (cwd_typed["cwdOptions"] or []),
    "cwd_start_field_was_browse_cleared": cwd_before_start["cwdValue"] == "",
    "cwd_start_launched_the_restored_path": (last_cwd_record.get("cwd") or last_cwd_record.get("requested_cwd") or "").strip() == "/tmp",
}

report = {
    "checks": checks,
    "observations": {
        "initial": initial,
        "pi_configured_pairs": pi_pairs,
        "browse_option_count": browse["optionCount"],
        "browse_group_labels": browse["groups"],
        "browse_active": browse["active"],
        "typed": replaced["value"],
        "typed_option_count": p["04-replace-on-type"]["optionCount"],
        "chosen": chosen,
        "launch_records": len(records),
        "last_launch": {k: last_record.get(k) for k in ("launch_id", "state", "model", "model_provider", "provider_choice", "agent_backend")},
        "cwd_recent_cwds": sorted(recent_cwds),
        "cwd_initial": cwd_initial["cwdValue"],
        "cwd_browse_options": cwd_browse["cwdOptions"],
        "cwd_typed_options": cwd_typed["cwdOptions"],
        "cwd_last_launch": {k: last_cwd_record.get(k) for k in ("launch_id", "state", "cwd", "requested_cwd")},
        "after_start_dialog_open": p["14-after-start"]["dialogOpen"],
    },
}
(artifacts / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print(f"artifacts: {artifacts}")
if failed:
    raise SystemExit("failed checks: " + ", ".join(failed))
PY

echo "new-session model combobox behavior verified; artifacts in $artifacts"
