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
  --mount "type=bind,src=$home_dir,dst=/home/tester" --workdir /workspace "$image" \
  python3 -m codoxear.server > "$artifacts/container-id.txt" || fail "container start failed"
wait_for_server || fail "server readiness check failed"

# Configured Pi pairs, straight from the API the dialog builds its list from.
curl -fsS -c "$root/cookies.txt" -H 'Content-Type: application/json' \
  --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"
curl -fsS -b "$root/cookies.txt" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions.json" || fail "could not read /api/sessions"
python3 - "$artifacts/sessions.json" > "$artifacts/expected-counts.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
defaults = (payload.get("new_session_defaults") or {}).get("backends") or {}

def counts(backend):
    entry = defaults.get(backend) or {}
    provider_models = entry.get("provider_models") or {}
    pairs = {(provider, model) for provider, models in provider_models.items() if isinstance(models, list) for model in models}
    if not pairs and not provider_models:
        return {"pairs": 0, "fallback": True}
    return {"pairs": len(pairs), "fallback": False}

print(json.dumps({"pi": counts("pi"), "codex": counts("codex"), "default_backend": payload.get("default_backend")}, indent=2))
PY

PROBE='(() => {
  const q = (s) => document.querySelector(s);
  const input = q("#newSessionModelInput"), menu = q("#newSessionModelMenu"), viewer = q("#newSessionViewer");
  const options = menu ? [...menu.querySelectorAll("[role=option]")] : [];
  const activeId = input ? input.getAttribute("aria-activedescendant") : null;
  const activeEl = activeId ? document.getElementById(activeId) : null;
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
open_dialog() { browser click '#chatEmptyNewBtn' >/dev/null 2>&1 || fail "could not open the new-session dialog"; wait_for_dialog || fail "new-session dialog never showed a committed model pair"; settle; }

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
  if "${docker[@]}" exec "$container" sh -lc 'test -s "$HOME/.local/share/codoxear/session_launches.jsonl"' >/dev/null 2>&1; then break; fi
  sleep 1
done
settle 600
probe 14-after-start

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

def model_of(display):
    return (display or "").split(" · ", 1)[0].strip()

def provider_of(display):
    parts = (display or "").split(" · ", 1)
    return parts[1].strip() if len(parts) > 1 else ""

checks = {
    "dialog_opens_on_a_committed_pair": p["01-initial"]["dialogOpen"] and bool(initial) and p["01-initial"]["menuDisplay"] == "none",
    "click_clears_the_committed_pair": browse["value"] == "",
    "click_opens_the_listbox": browse["menuDisplay"] == "block" and browse["expanded"] == "true",
    "browse_lists_every_configured_model": pi_pairs > 0 and browse["optionCount"] == pi_pairs,
    "browse_is_not_capped_at_twelve": browse["optionCount"] > 12 if pi_pairs > 12 else browse["optionCount"] == pi_pairs,
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
