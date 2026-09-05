#!/usr/bin/env bash
# Docker-isolated behavioral verifier for the theme engine and Settings dialog.
#
# Same isolation contract as scripts/docker_verify.sh: the app is built from a
# Git archive of the requested commit, runs under a container-only HOME with a
# fresh offline Pi session, and is driven through the host's agent-browser on a
# loopback port that is never the live 8743. This script exercises the real
# Settings dialog: family/mode switches, persistence across reload, custom CSS,
# reset, follow-system resolution, Escape/hint-mode policy, and captures a
# screenshot of every family x mode variant for design review.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_theme_verify.sh [commit]

Results (report.json plus one PNG per step/variant) are written to a new
/tmp/codoxear-theme-verify.* directory.

Environment:
  CODOXEAR_THEME_VERIFY_PORT       Loopback port (default: 19663; never 8743)
  CODOXEAR_THEME_VERIFY_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_THEME_VERIFY_PASSWORD   Login password (default: docker-theme-password)
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
port="${CODOXEAR_THEME_VERIFY_PORT:-19663}"
password="${CODOXEAR_THEME_VERIFY_PASSWORD:-docker-theme-password}"
image="${CODOXEAR_THEME_VERIFY_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-theme-verify-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-theme-verify-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-theme-verify.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-theme-verify-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_THEME_VERIFY_PORT must be a TCP port, got: $port" >&2
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

capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
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
    if [[ "$code" == "401" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_pi_session() {
  local cookie="$root/cookies.txt"
  curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
    --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"
  local sessions="$artifacts/sessions.json"
  for _ in $(seq 1 120); do
    if [[ "$(curl -sS -b "$cookie" -o "$sessions" -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions" 2>/dev/null || true)" == "200" ]] \
      && python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if any(isinstance(s,dict) and s.get("agent_backend")=="pi" for s in p.get("sessions",[])) else 1)' "$sessions"; then
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
  "$image" \
  bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id docker-theme-session -p "Initialize isolated theme verification session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-theme-session.jsonl" -print -quit)"
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
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "server readiness check failed"
wait_for_pi_session || fail "Pi session creation/discovery failed"

# Theme surface probe: what the sole writer is expected to have rendered.
THEME_PROBE='(() => {
  const html = document.documentElement;
  const link = document.getElementById("codoxearThemeLink");
  const style = document.getElementById("codoxearCustomCss");
  const meta = document.querySelector("meta[name=theme-color]");
  const links = [...document.querySelectorAll("link[rel=stylesheet]")].map((l) => l.getAttribute("href"));
  const cs = getComputedStyle(document.body);
  const dialog = document.getElementById("settingsViewer");
  const active = (sel) => { const n = document.querySelector(sel); return n ? n.classList.contains("active") : null; };
  return {
    theme: html.getAttribute("data-theme"), mode: html.getAttribute("data-mode"),
    href: link ? link.getAttribute("href") : null,
    linkOrder: links,
    customCss: style ? style.textContent : null,
    metaColor: meta ? meta.getAttribute("content") : null,
    bodyBg: cs.backgroundColor, bodyColor: cs.color, bodyFont: cs.fontFamily,
    sessionRadius: (() => { const s = document.querySelector("#sessions .session"); return s ? getComputedStyle(s).borderRadius : null; })(),
    sidebarBg: (() => { const s = document.querySelector(".sidebar"); return s ? getComputedStyle(s).backgroundColor : null; })(),
    dialogOpen: Boolean(dialog && dialog.open),
    swatchMode: (() => { const s = document.querySelector(".themeSwatches"); return s ? s.getAttribute("data-swatch-mode") : null; })(),
    activeFamily: ["paper","clay","slate"].find((f) => active(`[data-theme-family="${f}"]`)) || null,
    activeMode: ["system","light","dark"].find((m) => active(`[data-theme-mode="${m}"]`)) || null,
    modeHint: (() => { const h = document.getElementById("settingsModeHint"); return h ? h.textContent : null; })(),
    textarea: (() => { const t = document.getElementById("settingsCustomCss"); return t ? t.value : null; })(),
    hintBadges: document.querySelectorAll(".codoxear-hint-badge").length,
    storage: { family: localStorage.getItem("codoxear.ui.theme.family"), mode: localStorage.getItem("codoxear.ui.theme.mode"), customCss: localStorage.getItem("codoxear.ui.customCss") },
  };
})()'

probe() {
  browser eval "$THEME_PROBE" --json > "$artifacts/$1.json" 2>&1 || fail "probe $1 failed"
}

shot() {
  browser screenshot "$artifacts/$1.png" > /dev/null 2>&1 || fail "screenshot $1 failed"
}

open_settings() {
  browser click '#settingsBtnSide' > /dev/null 2>&1 || fail "settings button click failed"
  browser wait 400 > /dev/null 2>&1
}

close_settings() {
  browser click '#settingsCloseBtn' > /dev/null 2>&1 || fail "settings close click failed"
  browser wait 300 > /dev/null 2>&1
}

choose() {
  # $1 family|mode, $2 value. Waits for the swapped stylesheet to load.
  browser click "[data-theme-$1=\"$2\"]" > /dev/null 2>&1 || fail "choose $1=$2 failed"
  browser wait 900 > /dev/null 2>&1
}

browser set viewport 1280 860 > /dev/null 2>&1 || true
browser set media light > /dev/null 2>&1 || true
browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open app"
browser errors --clear --json > /dev/null 2>&1 || true
browser console --clear --json > /dev/null 2>&1 || true
browser fill '#pw' "$password" > /dev/null 2>&1 || fail "login password field was not available"
browser click '#loginBtn' > /dev/null 2>&1 || fail "login button was not available"
browser wait 3500 > /dev/null 2>&1
browser click '#sessions .session' > /dev/null 2>&1 || true
browser wait 1200 > /dev/null 2>&1

probe 01-initial
shot 01-paper-light-app

open_settings
probe 02-settings-open
shot 02-settings-dialog-paper-light
# Escape never closes a dialog; hint mode covers the dialog's own buttons.
browser press Escape > /dev/null 2>&1 || true
browser wait 200 > /dev/null 2>&1
probe 03-after-escape
browser press f > /dev/null 2>&1 || true
browser wait 250 > /dev/null 2>&1
probe 04-hint-mode
shot 04-settings-hint-mode
browser press Escape > /dev/null 2>&1 || true
browser wait 200 > /dev/null 2>&1

choose family slate
choose mode dark
probe 05-slate-dark-selected
shot 05-settings-dialog-slate-dark
close_settings
probe 06-slate-dark-app
shot 06-slate-dark-app

browser reload > /dev/null 2>&1 || fail "reload failed"
browser wait 3500 > /dev/null 2>&1
browser click '#sessions .session' > /dev/null 2>&1 || true
browser wait 800 > /dev/null 2>&1
probe 07-after-reload
shot 07-slate-dark-after-reload

# Custom CSS applies live (debounced) and survives the dialog closing.
open_settings
browser fill '#settingsCustomCss' '.topbar { outline: 3px solid rgb(255, 0, 128); }' > /dev/null 2>&1 || fail "custom css fill failed"
browser wait 700 > /dev/null 2>&1
browser eval '(() => getComputedStyle(document.querySelector(".topbar")).outlineColor)()' --json > "$artifacts/08-custom-css-outline.json" 2>&1 || fail "custom css probe failed"
probe 08-custom-css
shot 08-settings-custom-css
close_settings

# Remaining variants for design review: clay light/dark, paper dark, slate light.
open_settings
choose family clay
choose mode light
close_settings
shot 09-clay-light-app
open_settings
choose mode dark
close_settings
shot 10-clay-dark-app
open_settings
choose family paper
close_settings
shot 11-paper-dark-app
open_settings
choose family slate
choose mode light
close_settings
shot 12-slate-light-app

# Reset returns to paper + system and clears custom CSS.
open_settings
browser click '#settingsResetAppearanceBtn' > /dev/null 2>&1 || fail "reset click failed"
browser wait 900 > /dev/null 2>&1
probe 13-after-reset
shot 13-settings-after-reset
close_settings

# Follow-system: emulate a dark OS scheme and expect live re-resolution.
browser set media dark > /dev/null 2>&1 || fail "media emulation unavailable"
browser wait 900 > /dev/null 2>&1
probe 14-system-dark
shot 14-paper-system-dark-app
browser set media light > /dev/null 2>&1 || true
browser wait 900 > /dev/null 2>&1
probe 15-system-light

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true
capture_container_diagnostics

python3 - "$artifacts" <<'PY'
import json
import re
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])

def load(name):
    payload = json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise SystemExit(f"agent-browser {name} returned an unsuccessful response: {payload!r}")
    return payload["data"]["result"]

p = {name: load(name) for name in [
    "01-initial", "02-settings-open", "03-after-escape", "04-hint-mode", "05-slate-dark-selected",
    "06-slate-dark-app", "07-after-reload", "08-custom-css", "13-after-reset", "14-system-dark", "15-system-light",
]}
outline = load("08-custom-css-outline")
version = re.search(r"\?v=([0-9a-f]+)$", str(p["01-initial"]["href"] or "")) 
checks = {
    "boot_renders_paper_light": p["01-initial"]["theme"] == "paper" and p["01-initial"]["mode"] == "light",
    "theme_link_is_versioned_and_after_app_css": bool(version) and p["01-initial"]["linkOrder"][-1].startswith("themes/paper.css") and any(h.startswith("app.css") for h in p["01-initial"]["linkOrder"][:-1]),
    "meta_theme_color_paper_light": p["01-initial"]["metaColor"] == "#ffffff",
    "paper_light_is_square": p["01-initial"]["sessionRadius"] == "0px",
    "settings_opens_as_native_modal": p["02-settings-open"]["dialogOpen"] is True and p["02-settings-open"]["activeFamily"] == "paper" and p["02-settings-open"]["activeMode"] == "system",
    "escape_keeps_settings_open": p["03-after-escape"]["dialogOpen"] is True,
    "hint_mode_covers_dialog_buttons": p["04-hint-mode"]["hintBadges"] >= 8,
    "slate_dark_applies_live": p["05-slate-dark-selected"]["theme"] == "slate" and p["05-slate-dark-selected"]["mode"] == "dark" and p["05-slate-dark-selected"]["href"].startswith("themes/slate.css") and p["05-slate-dark-selected"]["swatchMode"] == "dark",
    "slate_dark_repaints_body": p["05-slate-dark-selected"]["bodyBg"] == "rgb(33, 33, 33)" and p["05-slate-dark-selected"]["sidebarBg"] == "rgb(23, 23, 23)",
    "slate_dark_rounds_sessions": p["06-slate-dark-app"]["sessionRadius"] == "12px",
    "meta_theme_color_slate_dark": p["06-slate-dark-app"]["metaColor"] == "#171717",
    "storage_persists_choice": p["06-slate-dark-app"]["storage"]["family"] == "slate" and p["06-slate-dark-app"]["storage"]["mode"] == "dark",
    "reload_boots_into_slate_dark": p["07-after-reload"]["theme"] == "slate" and p["07-after-reload"]["mode"] == "dark" and p["07-after-reload"]["bodyBg"] == "rgb(33, 33, 33)" and p["07-after-reload"]["href"].startswith("themes/slate.css"),
    "reload_keeps_single_theme_link": sum(h.startswith("themes/") for h in p["07-after-reload"]["linkOrder"]) == 1,
    "custom_css_applies_live": outline == "rgb(255, 0, 128)" and p["08-custom-css"]["storage"]["customCss"] is not None,
    "reset_restores_defaults": p["13-after-reset"]["theme"] == "paper" and p["13-after-reset"]["activeMode"] == "system" and p["13-after-reset"]["textarea"] == "" and p["13-after-reset"]["storage"] == {"family": None, "mode": None, "customCss": None},
    "system_mode_follows_dark_scheme": p["14-system-dark"]["mode"] == "dark" and p["14-system-dark"]["theme"] == "paper" and p["14-system-dark"]["bodyBg"] == "rgb(24, 22, 19)",
    "system_mode_returns_to_light": p["15-system-light"]["mode"] == "light",
}
errors = json.loads((artifacts / "browser-errors.json").read_text(encoding="utf-8")).get("data", {}).get("errors", [])
checks["no_page_errors"] = errors == []
summary = {"pass": all(checks.values()), "checks": checks, "probes": p, "customCssOutline": outline, "pageErrors": errors}
(artifacts / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pass": summary["pass"], "checks": checks}, indent=2))
if not summary["pass"]:
    raise SystemExit(1)
PY

printf 'PASS: commit=%s artifacts=%s\n' "$target_commit" "$artifacts"
