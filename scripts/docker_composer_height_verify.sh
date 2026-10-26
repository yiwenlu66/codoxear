#!/usr/bin/env bash
# Docker-isolated behavioral verifier for the composer height modes
# (normal / tall / full, the #expandBtn cycle + drag handle, per-session
# persistence, and the send-reset).
#
# Same isolation contract as scripts/docker_theme_verify.sh: the app is built
# from a Git archive of the requested commit, runs under a container-only HOME
# with two fresh offline Pi sessions, and is driven through the host's
# agent-browser on a loopback port that is never the live 8743. Every flow
# runs through the real composer: a real keyboard-less fill, real mouse clicks
# and a real mouse drag on the grabber strip (#expandBtn), plus synthetic PointerEvents for
# the deterministic drag/click-suppression and snap paths.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_composer_height_verify.sh [commit]

Results (report.json, one JSON probe per step, and the screenshots) are written
to a new /tmp/codoxear-composer-height-verify.* directory.

Environment:
  CODOXEAR_HEIGHT_VERIFY_PORT       Loopback port (default: 19664; never 8743)
  CODOXEAR_HEIGHT_VERIFY_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_HEIGHT_VERIFY_PASSWORD   Login password (default: docker-height-password)
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
port="${CODOXEAR_HEIGHT_VERIFY_PORT:-19664}"
password="${CODOXEAR_HEIGHT_VERIFY_PASSWORD:-docker-height-password}"
image="${CODOXEAR_HEIGHT_VERIFY_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-composer-height-verify-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-composer-height-verify-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-composer-height-verify.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-composer-height-verify-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_HEIGHT_VERIFY_PORT must be a TCP port, got: $port" >&2
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

wait_for_pi_sessions() {
  local cookie="$root/cookies.txt"
  curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
    --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"
  local sessions="$artifacts/sessions.json"
  for _ in $(seq 1 120); do
    if [[ "$(curl -sS -b "$cookie" -o "$sessions" -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions" 2>/dev/null || true)" == "200" ]] \
      && python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if sum(1 for s in p.get("sessions",[]) if isinstance(s,dict) and s.get("agent_backend")=="pi")>=2 else 1)' "$sessions"; then
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
# The browser loads dist/app.bundle.js, and the tracked copy is only rebuilt by
# deploy. Rebuild it from this commit's sources in the archive context so the
# verifier exercises the commit under test.
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"
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
    for sid in docker-height-a docker-height-b; do
      pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session-id "$sid" -p "Initialize isolated composer-height verification session." \
        >"$HOME/pi-bootstrap-$sid.log" 2>&1 || true
      session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*${sid}.jsonl" -print -quit)"
      test -n "$session_log"
      CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
        codoxear-broker --cwd /workspace -- \
          --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
          --session "$session_log" >"$HOME/sessiond-$sid.log" 2>&1 &
    done
    for _ in $(seq 1 120); do
      n=0
      for meta in "$HOME"/.local/share/codoxear/socks/*.json; do
        if test -f "$meta" && grep -q session_id "$meta"; then n=$((n+1)); fi
      done
      if test "$n" -ge 2; then break; fi
      sleep 1
    done
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "server readiness check failed"
wait_for_pi_sessions || fail "Pi session creation/discovery failed (two sessions required)"

# Composer state probe. The interrupt button is only shown while a turn is
# running, so its reachability is tested by temporarily forcing it visible and
# hit-testing its center, then restoring the previous display value.
PROBE='(() => {
  const c = document.querySelector(".composer");
  const t = document.getElementById("msg");
  const b = document.getElementById("expandBtn");
  const cw = document.getElementById("chatWrap");
  const tb = document.querySelector(".topbar");
  const ib = document.getElementById("interruptBtn");
  const r = (n) => { if (!n) return null; const x = n.getBoundingClientRect(); return { top: x.top, bottom: x.bottom, left: x.left, right: x.right, width: x.width, height: x.height }; };
  const appH = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--appH")) || innerHeight;
  const active = document.querySelector("#sessions .session.active");
  const sid = active ? active.dataset.sessionId : null;
  const keys = Object.keys(localStorage).filter((k) => k.startsWith("codexweb.composerHeight."));
  let interruptReachable = null;
  if (ib) {
    const prev = ib.style.display;
    ib.style.display = "inline-flex";
    const ir = ib.getBoundingClientRect();
    const hit = document.elementFromPoint(ir.left + ir.width / 2, ir.top + ir.height / 2);
    interruptReachable = Boolean(hit && (hit === ib || ib.contains(hit)) && ir.width > 0 && ir.top >= 0 && ir.bottom <= innerHeight);
    ib.style.display = prev;
  }
  const bd = b ? getComputedStyle(b).display : null;
  const pill = document.querySelector("#expandBtn .composerGrabPill");
  const pillBg = pill ? getComputedStyle(pill).backgroundColor : null;
  // Resolve a custom property to the browser-normalized rgb() string by
  // painting it on a throwaway element, so it is comparable to computed colors.
  const resolveVar = (name) => { const probeEl = document.createElement("span"); probeEl.style.backgroundColor = "var(" + name + ")"; document.body.appendChild(probeEl); const v = getComputedStyle(probeEl).backgroundColor; probeEl.remove(); return v; };
  const form = c ? c.querySelector("form") : null;
  const row = document.querySelector(".composerInputRow");
  const wrap = document.querySelector(".composer .inputWrap");
  const clear = document.getElementById("clearBtn");
  const strip = document.querySelector(".composerStrip");
  return {
    sid, appH, innerHeight,
    height: c ? c.dataset.height : null,
    controlLayout: c ? c.dataset.controls || null : null,
    clearDisplay: clear ? getComputedStyle(clear).display : null,
    tallVar: c ? c.style.getPropertyValue("--composer-tall-h") : null,
    expandDisplay: bd,
    expandVisible: Boolean(b && bd !== "none" && b.getBoundingClientRect().height > 0),
    expandLabel: b ? b.getAttribute("aria-label") : null,
    expandPressed: b ? b.getAttribute("aria-pressed") : null,
    textareaH: t ? t.getBoundingClientRect().height : null,
    textareaComputedH: t ? getComputedStyle(t).height : null,
    draftLines: t ? t.value.split("\n").length : null,
    chatWrapDisplay: cw ? getComputedStyle(cw).display : null,
    chatWrapH: cw ? cw.getBoundingClientRect().height : null,
    topbar: r(tb), topbarVisible: Boolean(tb && getComputedStyle(tb).display !== "none" && tb.getBoundingClientRect().height > 0),
    interruptReachable,
    composer: r(c), textarea: r(t), expandRect: r(b),
    formRect: r(form), inputRowRect: r(row), inputWrapRect: r(wrap), stripRect: r(strip),
    expandInInputRow: Boolean(b && b.closest(".composerInputRow")),
    expandInStrip: Boolean(b && strip && b.parentElement === strip),
    stripParentIsForm: Boolean(strip && form && strip.parentElement === form),
    stripIsFirstFormChild: Boolean(strip && form && form.firstElementChild === strip),
    expandCursor: b ? getComputedStyle(b).cursor : null,
    expandTouchAction: b ? getComputedStyle(b).touchAction : null,
    pillBg, pillRect: r(pill), accentRgb: resolveVar("--accent"), borderRgb: resolveVar("--border"),
    sendRect: r(document.getElementById("sendBtn")), attachRect: r(document.getElementById("attachBtn")), queueRect: r(document.getElementById("queueBtn")),
    clearRect: r(clear),
    storageKeys: keys,
    storedForSession: sid ? localStorage.getItem("codexweb.composerHeight." + sid) : null,
    storedAll: Object.fromEntries(keys.map((k) => [k, localStorage.getItem(k)])),
  };
})()'

probe() {
  browser eval "$PROBE" --json > "$artifacts/$1.json" 2>&1 || fail "probe $1 failed"
}
shot() {
  browser screenshot "$artifacts/$1.png" > /dev/null 2>&1 || fail "screenshot $1 failed"
}
settle() { browser wait "${1:-450}" > /dev/null 2>&1; }
click_expand() {
  browser click '#expandBtn' > /dev/null 2>&1 || fail "expand button click failed"
  settle
}
select_session() {
  browser click "#sessions .session[data-session-id=\"$1\"]" > /dev/null 2>&1 || fail "could not select session $1"
  settle 1200
}
session_ids() {
  browser eval '(() => [...document.querySelectorAll("#sessions .session[data-session-id]")].map((n) => n.dataset.sessionId))()' --json > "$artifacts/session-ids.json" 2>&1 || fail "could not list sessions"
  python3 - "$artifacts/session-ids.json" <<'PY'
import json, sys
ids = json.load(open(sys.argv[1]))["data"]["result"]
if len(ids) < 2:
    raise SystemExit("fewer than two sessions in the sidebar: %r" % ids)
print(ids[0], ids[1])
PY
}
expand_center() {
  browser eval '(() => { const r = document.getElementById("expandBtn").getBoundingClientRect(); return [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)]; })()' --json > "$artifacts/expand-center.json" 2>&1 || fail "expand center probe failed"
  python3 -c 'import json,sys; x,y=json.load(open(sys.argv[1]))["data"]["result"]; print(x,y)' "$artifacts/expand-center.json"
}
# Synthetic drag on #expandBtn: pointerdown at the button center, then
# pointermoves at (start - d) for each d in $1 (space-separated, negative d
# drags down), then pointerup at the last position.
synthetic_drag() {
  local deltas="$1"
  browser eval "(() => {
    const b = document.getElementById('expandBtn');
    const r = b.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const ev = (type, cy) => new PointerEvent(type, { pointerId: 7, pointerType: 'mouse', button: 0, buttons: type === 'pointerup' ? 0 : 1, clientX: x, clientY: cy, bubbles: true, cancelable: true, isPrimary: true });
    b.dispatchEvent(ev('pointerdown', y));
    let last = y;
    for (const d of [${deltas// /,}]) { last = y - d; b.dispatchEvent(ev('pointermove', last)); }
    b.dispatchEvent(ev('pointerup', last));
    return true;
  })()" --json > /dev/null 2>&1 || fail "synthetic drag failed"
  settle
}

LONG_DRAFT="$(for i in $(seq 1 40); do printf 'line %02d of a deliberately long composer draft\n' "$i"; done)"
LONG_DRAFT="${LONG_DRAFT%$'\n'}"

browser set viewport 1280 860 > /dev/null 2>&1 || true
browser set media light > /dev/null 2>&1 || true
browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open app"
browser errors --clear --json > /dev/null 2>&1 || true
browser console --clear --json > /dev/null 2>&1 || true
browser fill '#pw' "$password" > /dev/null 2>&1 || fail "login password field was not available"
browser click '#loginBtn' > /dev/null 2>&1 || fail "login button was not available"
browser wait 3500 > /dev/null 2>&1

read -r SID_A SID_B < <(session_ids)
printf '%s\n%s\n' "$SID_A" "$SID_B" > "$artifacts/session-order.txt"
select_session "$SID_A"

# a. short draft keeps the button hidden
browser eval '(() => { for (const k of Object.keys(localStorage)) if (k.startsWith("codexweb.composerHeight.")) localStorage.removeItem(k); return true; })()' --json > /dev/null 2>&1 || true
browser fill '#msg' 'hello' > /dev/null 2>&1 || fail "could not type a short draft"
settle
probe a-short-draft

# a2. a four-line draft has room for the stacked control column (and Clear) while
# the draft still fits the normal cap, so the grabber is not involved.
DRAFT_TWO_LINES=$'first line\nsecond line'
DRAFT_THREE_LINES=$'one\ntwo\nthree'
DRAFT_FOUR_LINES=$'one\ntwo\nthree\nfour'
browser fill '#msg' "$DRAFT_TWO_LINES" > /dev/null 2>&1 || fail "could not type a two-line draft"
settle
probe a2-two-lines
browser fill '#msg' "$DRAFT_THREE_LINES" > /dev/null 2>&1 || fail "could not type a three-line draft"
settle
probe a3-three-lines
browser fill '#msg' "$DRAFT_FOUR_LINES" > /dev/null 2>&1 || fail "could not type a four-line draft"
settle
probe a4-four-lines
shot 00-clear-without-grabber

# b. long draft: button appears, normal textarea stays capped
browser fill '#msg' "$LONG_DRAFT" > /dev/null 2>&1 || fail "could not type a long draft"
settle
probe b-long-draft
shot 01-normal-long-draft

# c. click -> tall
click_expand
probe c-tall
shot 02-tall

# d. click -> full
click_expand
probe d-full
shot 03-full

# e. click -> normal
click_expand
probe e-normal-again

# f. persistence: tall survives reload and a session round trip
click_expand
probe f1-tall-before-reload
browser reload > /dev/null 2>&1 || fail "reload failed"
browser wait 3500 > /dev/null 2>&1
select_session "$SID_A"
probe f2-tall-after-reload
select_session "$SID_B"
probe f3-other-session
select_session "$SID_A"
probe f4-back-on-first

# g. send resets the mode and drops the key
browser eval '(() => { document.getElementById("msg").focus(); return true; })()' --json > /dev/null 2>&1 || true
probe g1-before-send
browser press Control+Enter > /dev/null 2>&1 || fail "Ctrl+Enter failed"
browser wait 2500 > /dev/null 2>&1
probe g2-after-send
shot 04-after-send-reset

# h. drag. h1: real mouse drag from normal; the click that follows pointerup
# on a real mouse must be suppressed, so the mode stays tall.
browser fill '#msg' "$LONG_DRAFT" > /dev/null 2>&1 || fail "could not retype the long draft"
settle
probe h0-normal-long-again
read -r BX BY < <(expand_center)
browser mouse move "$BX" "$BY" > /dev/null 2>&1 || fail "mouse move failed"
browser mouse down > /dev/null 2>&1 || fail "mouse down failed"
for d in 20 100 200 300; do
  browser mouse move "$BX" "$((BY - d))" > /dev/null 2>&1 || fail "mouse drag move failed"
done
browser mouse up > /dev/null 2>&1 || fail "mouse up failed"
settle
probe h1-real-drag-up
shot 05-real-drag-tall
# Persisted dragged height survives a reload.
browser reload > /dev/null 2>&1 || fail "reload after drag failed"
browser wait 3500 > /dev/null 2>&1
select_session "$SID_A"
probe h1b-dragged-after-reload

# h2. synthetic drag down 100px from tall: still tall, smaller; the click right
# after pointerup is swallowed once; the next click cycles tall -> full.
synthetic_drag "20 50 100 -100"
probe h2-synthetic-drag-down
browser eval '(() => { document.getElementById("expandBtn").click(); return true; })()' --json > /dev/null 2>&1 || fail "post-drag click failed"
settle
probe h2b-click-after-drag-suppressed
browser eval '(() => { document.getElementById("expandBtn").click(); return true; })()' --json > /dev/null 2>&1 || fail "second click failed"
settle
probe h2c-next-click-cycles
# h3. full -> normal by click, then drag far past 0.9 x app height -> full.
click_expand
probe h3a-back-to-normal
synthetic_drag "20 100 400 900"
probe h3b-drag-past-full
shot 06-drag-full
# h4. drag far down from full snaps to normal.
synthetic_drag "-20 -300 -9000"
probe h4-drag-down-normal

# i. preview isolation: a stored committed tall height (500) must survive an
# aborted drag that previews other tall heights and lands on normal; after the
# suppressed click, the next click re-enters tall at the committed 500px.
browser eval "(() => { localStorage.setItem('codexweb.composerHeight.${SID_A}', JSON.stringify({mode:'tall',height:500})); return true; })()" --json > /dev/null 2>&1 || fail "could not seed committed tall height"
browser reload > /dev/null 2>&1 || fail "reload before isolation probe failed"
browser wait 3500 > /dev/null 2>&1
select_session "$SID_A"
browser fill '#msg' "$LONG_DRAFT" > /dev/null 2>&1 || fail "could not retype the long draft"
settle
probe i0-seeded-tall
synthetic_drag "-100 -300 -9000"
probe i1-aborted-drag-lands-normal
browser eval '(() => { document.getElementById("expandBtn").click(); return true; })()' --json > /dev/null 2>&1 || fail "isolation click 1 failed"
settle
probe i2-first-click-suppressed
browser eval '(() => { document.getElementById("expandBtn").click(); return true; })()' --json > /dev/null 2>&1 || fail "isolation click 2 failed"
settle
probe i3-second-click-reuses-committed

# j. phone viewport: the same room rule and centering at 390x844. The layout
# switch happens in place (the sidebar collapses into a drawer, so a session
# re-selection would need the drawer opened first); the height mode is cycled
# back to normal instead.
browser set viewport 390 844 > /dev/null 2>&1 || true
settle 900
browser eval '(() => { const c = document.querySelector(".composer"); let n = 0; while (c.dataset.height !== "normal" && n < 4) { document.getElementById("expandBtn").click(); n += 1; } return c.dataset.height; })()' --json > "$artifacts/phone-normal-mode.json" 2>&1 || fail "could not return the phone composer to normal mode"
settle
browser fill '#msg' "$DRAFT_FOUR_LINES" > /dev/null 2>&1 || fail "could not type the four-line draft on the phone viewport"
settle
probe j1-phone-four-lines
shot 07-phone-four-lines

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true
capture_container_diagnostics

python3 - "$artifacts" <<'PY'
import json
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])

def load(name):
    payload = json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise SystemExit(f"agent-browser {name} returned an unsuccessful response: {payload!r}")
    return payload["data"]["result"]

names = [
    "a-short-draft", "a2-two-lines", "a3-three-lines", "a4-four-lines", "b-long-draft", "c-tall", "d-full", "e-normal-again",
    "f1-tall-before-reload", "f2-tall-after-reload", "f3-other-session", "f4-back-on-first",
    "g1-before-send", "g2-after-send", "h0-normal-long-again", "h1-real-drag-up", "h1b-dragged-after-reload",
    "h2-synthetic-drag-down", "h2b-click-after-drag-suppressed", "h2c-next-click-cycles",
    "h3a-back-to-normal", "h3b-drag-past-full", "h4-drag-down-normal",
    "i0-seeded-tall", "i1-aborted-drag-lands-normal", "i2-first-click-suppressed", "i3-second-click-reuses-committed",
    "j1-phone-four-lines",
]
p = {n: load(n) for n in names}
sid_a = (artifacts / "session-order.txt").read_text().split()[0]
key_a = f"codexweb.composerHeight.{sid_a}"

def stored(probe):
    raw = probe["storedAll"].get(key_a)
    return json.loads(raw) if raw else None

def near(value, target, tol):
    return abs(value - target) <= tol * target

a, b, c, d, e = (p[n] for n in ["a-short-draft", "b-long-draft", "c-tall", "d-full", "e-normal-again"])
f2, f3, f4 = p["f2-tall-after-reload"], p["f3-other-session"], p["f4-back-on-first"]
g1, g2 = p["g1-before-send"], p["g2-after-send"]
h1, h1b, h2, h2b, h2c = (p[n] for n in ["h1-real-drag-up", "h1b-dragged-after-reload", "h2-synthetic-drag-down", "h2b-click-after-drag-suppressed", "h2c-next-click-cycles"])
h3a, h3b, h4 = p["h3a-back-to-normal"], p["h3b-drag-past-full"], p["h4-drag-down-normal"]

below_topbar = lambda x: x["appH"] - x["topbar"]["bottom"]
checks = {
    "a_short_draft_hides_expand_button": a["height"] == "normal" and a["expandDisplay"] == "none" and a["expandVisible"] is False,
    "b_long_draft_shows_expand_button": b["draftLines"] >= 30 and b["expandVisible"] is True and b["height"] == "normal",
    "b_normal_textarea_stays_capped_at_180": b["textareaH"] <= 181,
    "c_click_enters_tall": c["height"] == "tall" and c["expandPressed"] == "true",
    "c_tall_textarea_is_55_percent_of_app_height": near(c["textareaH"], 0.55 * c["appH"], 0.10),
    "c_tall_keeps_transcript_visible": c["chatWrapDisplay"] != "none" and c["chatWrapH"] > 50,
    "c_tall_persists_mode_only": stored(c) == {"mode": "tall"},
    "d_second_click_enters_full": d["height"] == "full" and d["expandLabel"] == "Restore draft height",
    "d_full_hides_transcript": d["chatWrapDisplay"] == "none",
    "d_full_textarea_fills_below_topbar": d["textareaH"] >= 0.75 * below_topbar(d) and d["textarea"]["top"] >= d["topbar"]["bottom"] and d["composer"]["bottom"] <= d["appH"] + 1,
    "d_full_keeps_topbar_and_interrupt_reachable": d["topbarVisible"] is True and d["topbar"]["top"] >= 0 and d["interruptReachable"] is True,
    "d_full_persists_mode": stored(d) == {"mode": "full"},
    "e_third_click_returns_to_normal": e["height"] == "normal" and e["chatWrapDisplay"] != "none" and e["chatWrapH"] > 50,
    "e_normal_removes_storage_key": key_a not in e["storageKeys"],
    "f_reload_restores_tall_with_same_height": p["f1-tall-before-reload"]["height"] == "tall" and f2["height"] == "tall" and near(f2["textareaH"], p["f1-tall-before-reload"]["textareaH"], 0.02) and stored(f2) == {"mode": "tall"},
    "f_other_session_stays_normal_and_button_hidden": f3["sid"] != sid_a and f3["height"] == "normal" and f3["expandVisible"] is False,
    "f_return_to_first_session_is_tall_again": f4["sid"] == sid_a and f4["height"] == "tall" and near(f4["textareaH"], f2["textareaH"], 0.02),
    "g_precondition_tall_with_long_draft": g1["height"] == "tall" and g1["draftLines"] >= 30 and key_a in g1["storageKeys"],
    "g_send_resets_to_normal": g2["height"] == "normal" and g2["draftLines"] == 1,
    "g_send_removes_storage_key": key_a not in g2["storageKeys"],
    "g_send_hides_expand_button_for_empty_draft": g2["expandVisible"] is False,
    "h_real_mouse_drag_precondition": p["h0-normal-long-again"]["height"] == "normal" and p["h0-normal-long-again"]["expandVisible"] is True,
    "h_real_drag_up_ends_tall_with_custom_height": h1["height"] == "tall" and bool(h1["tallVar"]) and h1["tallVar"].endswith("px") and h1["textareaH"] > p["h0-normal-long-again"]["textareaH"] + 150,
    "h_real_drag_suppresses_following_click": h1["height"] == "tall",
    "h_dragged_height_persists_in_storage": (stored(h1) or {}).get("mode") == "tall" and isinstance((stored(h1) or {}).get("height"), (int, float)),
    "h_dragged_height_survives_reload": h1b["height"] == "tall" and near(h1b["textareaH"], h1["textareaH"], 0.02) and h1b["tallVar"] == h1["tallVar"],
    "h_synthetic_drag_down_shrinks_tall": h2["height"] == "tall" and h2["textareaH"] < h1b["textareaH"] - 50,
    "h_click_after_synthetic_drag_is_suppressed": h2b["height"] == "tall" and h2b["tallVar"] == h2["tallVar"],
    "h_suppression_is_one_shot": h2c["height"] == "full",
    "h_synthetic_drag_past_90_percent_snaps_full": h3a["height"] == "normal" and h3b["height"] == "full" and h3b["chatWrapDisplay"] == "none" and stored(h3b) == {"mode": "full"},
    "h_drag_down_from_full_snaps_normal": h4["height"] == "normal" and key_a not in h4["storageKeys"] and h4["chatWrapDisplay"] != "none",
}
i0, i1, i2, i3 = (p[n] for n in ["i0-seeded-tall", "i1-aborted-drag-lands-normal", "i2-first-click-suppressed", "i3-second-click-reuses-committed"])
checks["i_seed_loads_committed_tall_500"] = i0["height"] == "tall" and i0["tallVar"] == "500px"
checks["i_aborted_drag_lands_normal_keeps_committed_preference"] = i1["height"] == "normal"
checks["i_first_click_after_abort_suppressed"] = i2["height"] == "normal"
checks["i_second_click_reuses_committed_500px"] = i3["height"] == "tall" and i3["tallVar"] == "500px" and (json.loads(i3["storedAll"].get(key_a) or "null") == {"mode": "tall", "height": 500})
# Vertically centered controls: send centers on the row's cross axis (the
# textarea's middle) by itself, and the stacked control column centers as one
# group — its member buttons cannot each be centered, the column can.
def row_middle(x):
    ta = x["textarea"]
    return (ta["top"] + ta["bottom"]) / 2

def centered_controls(x):
    send = x["sendRect"]
    if not (send and send["height"] > 0):
        return False
    if abs((send["top"] + send["bottom"]) / 2 - row_middle(x)) > 2:
        return False
    column = [x[k] for k in ("clearRect", "attachRect", "queueRect") if x[k] and x[k]["height"] > 0]
    if not column:
        return False
    top = min(rect["top"] for rect in column)
    bottom = max(rect["bottom"] for rect in column)
    return abs((top + bottom) / 2 - row_middle(x)) <= 2
checks["overflow_controls_vertically_centered"] = centered_controls(b)
checks["tall_controls_vertically_centered"] = centered_controls(c)
checks["full_controls_vertically_centered"] = centered_controls(d)
checks["send_centered_clear_of_the_foot"] = bool(
    c["sendRect"] and c["textarea"]["bottom"] - c["sendRect"]["bottom"] > 20
    and abs((c["sendRect"]["top"] + c["sendRect"]["bottom"]) / 2 - (c["textarea"]["top"] + c["textarea"]["bottom"]) / 2) <= 2
)
# Room-based Clear: the control column needs two control heights (2 x 32px), the
# 12px stack gap and a 4px margin each side = 84px. A normal auto-grown draft
# reaches 72px at three lines and 92px at four, so Clear arrives at the fourth
# line — while the draft still fits the cap, i.e. before the grabber exists.
a2, a3, a4 = p["a2-two-lines"], p["a3-three-lines"], p["a4-four-lines"]
checks["two_line_draft_has_no_control_column"] = a2["controlLayout"] is None and a2["clearDisplay"] == "none" and a2["expandVisible"] is False
checks["three_line_draft_has_no_control_column"] = a3["controlLayout"] is None and a3["clearDisplay"] == "none" and a3["expandVisible"] is False
checks["four_line_draft_shows_clear_before_the_grabber"] = bool(
    a4["controlLayout"] == "stack" and a4["clearDisplay"] != "none" and a4["clearRect"] and a4["clearRect"]["height"] > 0
    and a4["expandVisible"] is False
)
# Grabber strip placement: a full-width strip (the form's content box, i.e. the
# same edges as the input row, covering >= 95% of the form's border box), first
# child of the form, above the textarea's top edge and outside the input row.
def strip_ok(x):
    s, f, t, row = x["stripRect"], x["formRect"], x["textarea"], x["inputRowRect"]
    return bool(
        x["expandVisible"] and s and f and row
        and abs(s["left"] - row["left"]) <= 1 and abs(s["right"] - row["right"]) <= 1
        and s["width"] >= 0.95 * f["width"]
        and s["bottom"] <= t["top"] + 0.5 and s["height"] > 0
        and x["expandInStrip"] and x["stripParentIsForm"] and x["stripIsFirstFormChild"]
        and not x["expandInInputRow"]
    )
for label, x in (("normal", b), ("tall", c), ("full", d)):
    checks[f"strip_{label}_spans_form_width_above_textarea_outside_input_row"] = strip_ok(x)
checks["strip_is_a_row_resize_touch_none_drag_surface"] = b["expandCursor"] == "row-resize" and b["expandTouchAction"] == "none"
checks["strip_pill_is_centered_and_narrow"] = bool(
    b["pillRect"] and abs((b["pillRect"]["left"] + b["pillRect"]["right"]) / 2 - (b["expandRect"]["left"] + b["expandRect"]["right"]) / 2) <= 1.5
    and 0 < b["pillRect"]["width"] <= 60 and 0 < b["pillRect"]["height"] <= 8
)
# Pill feedback: --border in normal, --accent in tall/full, back to --border after cycling to normal.
checks["pill_normal_uses_border_color"] = b["pillBg"] == b["borderRgb"]
checks["pill_tall_uses_accent_color"] = c["pillBg"] == c["accentRgb"] and c["pillBg"] != b["pillBg"]
checks["pill_full_uses_accent_color"] = d["pillBg"] == d["accentRgb"] and d["pillBg"] != b["pillBg"]
checks["pill_returns_to_border_color_in_normal"] = e["pillBg"] == e["borderRgb"]
# The grabber must cost no horizontal space: the input row and .inputWrap keep
# the same width when the draft crosses the overflow threshold (strip appears).
checks["strip_appearing_does_not_shrink_input_wrap"] = (
    a["expandVisible"] is False and b["expandVisible"] is True
    and b["inputWrapRect"]["width"] >= a["inputWrapRect"]["width"] - 0.5
    and abs(b["inputRowRect"]["width"] - a["inputRowRect"]["width"]) <= 0.5
    and abs(b["textarea"]["width"] - a["textarea"]["width"]) <= 0.5
)
j1 = p["j1-phone-four-lines"]
checks["phone_four_line_draft_shows_clear_without_the_grabber"] = bool(
    j1["controlLayout"] == "stack" and j1["clearDisplay"] != "none" and j1["expandVisible"] is False
)
checks["phone_controls_vertically_centered"] = centered_controls(j1)
errors = json.loads((artifacts / "browser-errors.json").read_text(encoding="utf-8")).get("data", {}).get("errors", [])
checks["no_page_errors"] = errors == []
summary = {"pass": all(checks.values()), "checks": checks, "probes": p, "pageErrors": errors}
(artifacts / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pass": summary["pass"], "checks": checks}, indent=2))
if not summary["pass"]:
    raise SystemExit(1)
PY

printf 'PASS: commit=%s artifacts=%s\n' "$target_commit" "$artifacts"
