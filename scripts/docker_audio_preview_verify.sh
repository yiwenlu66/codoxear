#!/usr/bin/env bash
# Docker-isolated real-browser verifier for audio playback in the file viewer.
# It builds the requested committed revision from a Git archive, creates a
# container-only Pi session rooted at a small Git fixture holding real media
# (ffmpeg is available in the verify image), and records every browser command
# and raw result under a fresh artifact directory.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_audio_preview_verify.sh [commit]

Behaviorally verifies, in a real Chromium:
  A. an MP3 opens in the audio surface and plays from the original blob URL,
  B. an Ogg Vorbis file (undecodable on the phone clients, so never trusted) is
     converted on demand and plays from the media_preview URL,
  C. closing the viewer stops playback and unloads the element,
  D. an MKV still converts to MP4 and plays in the video surface (regression).

Artifacts (raw browser command output, JSON assertions, screenshots, container
logs) are retained in /tmp/codoxear-audio-verify.*.

Environment:
  CODOXEAR_AUDIO_VERIFY_PORT       Loopback port (default: 19644; never 8743)
  CODOXEAR_AUDIO_VERIFY_IMAGE      Image tag override
  CODOXEAR_AUDIO_VERIFY_PASSWORD   Login password override
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
port="${CODOXEAR_AUDIO_VERIFY_PORT:-19644}"
password="${CODOXEAR_AUDIO_VERIFY_PASSWORD:-docker-audio-password}"
image="${CODOXEAR_AUDIO_VERIFY_IMAGE:-codoxear-audio-verify:${short_commit}}"
container="codoxear-audio-verify-${short_commit}-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-audio-verify-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-audio-verify.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-audio-verify-${short_commit}-$$"
readonly repo_root target_commit short_commit port password image container source_pi_agent root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_AUDIO_VERIFY_PORT must be a TCP port, got: $port" >&2
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

browser() { AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"; }
browser_step() { # artifact stem, followed by agent-browser command and args
  local stem="$1"
  shift
  printf '$ agent-browser' > "$artifacts/${stem}.txt"
  printf ' %q' "$@" >> "$artifacts/${stem}.txt"
  printf '\n' >> "$artifacts/${stem}.txt"
  browser "$@" >> "$artifacts/${stem}.txt" 2>&1
}

capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/container.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf -- "--- fixture ---\n"
    ls -l /workspace/demo
    printf -- "--- media previews ---\n"
    ls -l "$HOME/.local/share/codoxear/media_previews" 2>/dev/null || true
    printf -- "--- socks ---\n"
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -exec sh -c '\''echo --- "$1"; cat "$1"'\'' _ {} \; 2>/dev/null || true
  ' > "$artifacts/container-isolation.txt" 2>&1 || true
}

cleanup() {
  capture_container_diagnostics
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  sudo /bin/rm -rf "$root"
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
    [[ -r "$source_pi_agent/$name" ]] && install -m 600 "$source_pi_agent/$name" "$home_dir/.pi/agent/$name"
  done
}

wait_for_server() {
  local status
  for _ in $(seq 1 120); do
    status="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
    [[ "$status" == "401" ]] && return 0
    sleep 1
  done
  return 1
}

wait_for_pi_session() {
  local sessions="$artifacts/api-sessions.json" status
  curl -fsS -c "$root/cookies.txt" -H 'Content-Type: application/json' \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"password":sys.argv[1]}))' "$password")" \
    "http://127.0.0.1:${port}/api/login" > "$artifacts/api-login.json"
  for _ in $(seq 1 120); do
    status="$(curl -sS -b "$root/cookies.txt" -o "$sessions" -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions" 2>/dev/null || true)"
    if [[ "$status" == "200" ]] && python3 - "$sessions" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("sessions", [])
raise SystemExit(0 if any(isinstance(row, dict) and row.get("agent_backend") == "pi" and row.get("session_id") for row in rows) else 1)
PY
    then return 0; fi
    sleep 1
  done
  return 1
}

copy_pi_config

build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
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
  --user root \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    fixture=/workspace/demo
    mkdir -p "$fixture"
    cd "$fixture"
    git init -q
    git config user.email verify@example.test
    git config user.name "Codoxear verifier"
    # Real media, produced by the image’s own ffmpeg: the fixture is what a
    # session directory actually holds, not a hand-written byte blob.
    ffmpeg -hide_banner -loglevel error -f lavfi -i "sine=frequency=440:duration=1.2" \
      -c:a libmp3lame -q:a 4 tone.mp3
    ffmpeg -hide_banner -loglevel error -f lavfi -i "sine=frequency=330:duration=1.2" \
      -c:a libvorbis -q:a 4 speech.ogg
    ffmpeg -hide_banner -loglevel error -f lavfi -i "testsrc=size=160x120:rate=10" \
      -f lavfi -i "sine=frequency=220:duration=0.9" -t 0.9 \
      -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac clip.mkv
    printf "# Fixture\n" > README.md
    git add README.md tone.mp3 speech.ogg clip.mkv
    git commit -qm fixture
    cd /workspace
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id docker-audio-verify -p "Initialize isolated audio playback verification session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-audio-verify.jsonl" -print -quit)"
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
wait_for_pi_session || fail "offline Pi session did not appear"
session_id="$(python3 - "$artifacts/api-sessions.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8")).get("sessions", []):
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and isinstance(row.get("session_id"), str):
        print(row["session_id"])
        break
else: raise SystemExit("no Pi session")
PY
)"
printf '%s\n' "$session_id" > "$artifacts/session-id.txt"
card_selector="#sessions .session[data-session-id=\"${session_id}\"]"

app_url="http://127.0.0.1:${port}/"
# Autoplay policy is a verification-only launch argument: it lets the script
# drive the element the way a user's tap would, instead of reporting a rejected
# play() as a media failure.
browser_step 01-open open "$app_url" --args "--autoplay-policy=no-user-gesture-required" || fail "browser could not open app"
browser_step 02-errors-clear errors --clear --json || fail "cannot clear browser errors"
browser_step 03-console-clear console --clear --json || fail "cannot clear browser console"
browser_step 04-login-fill fill '#pw' "$password" || fail "login field missing"
browser_step 05-login-click click '#loginBtn' || fail "login button missing"
browser_step 06-login-wait wait 3500 || fail "login did not settle"
browser_step 07-select-session click "$card_selector" || fail "Pi session card missing"
browser_step 08-select-wait wait 1600 || fail "session selection did not settle"

# A: MP3 (a type every target client decodes) plays straight from the original.
browser_step 09-open-picker click '#fileBtn' || fail "file picker button missing"
browser_step 10-search-tone fill '#filePickerInput' 'tone.mp3' || fail "picker input missing"
browser_step 11-tone-options eval '(() => Array.from(document.querySelectorAll("#filePickerMenu .fileMenuItem")).map((n)=>({id:n.id,text:n.innerText})))()' --json || fail "cannot inspect tone.mp3 picker rows"
browser_step 12-open-tone click '#filePickerMenu .fileMenuItem' || fail "tone.mp3 picker row missing"
browser_step 13-tone-wait wait 2000 || fail "tone.mp3 did not settle"
browser_step 14-tone-surface eval '(async () => { const visible=(n)=>!!n&&getComputedStyle(n).display!=="none"&&n.getBoundingClientRect().width>0&&n.getBoundingClientRect().height>0; const audio=document.querySelector("#fileAudio"); const video=document.querySelector("#fileVideo"); const status=document.querySelector("#fileStatus"); const previewBtn=document.querySelector("#fileMediaPreviewBtn"); if (!audio) return {missing:true}; try { await audio.play(); } catch (e) { return {playError:String(e), src:audio.currentSrc||audio.getAttribute("src")}; } await new Promise((r)=>setTimeout(r,700)); return {src:audio.currentSrc||audio.getAttribute("src"), previewSrc:(audio.currentSrc||"").includes("/file/media_preview"), audioVisible:visible(audio), videoVisible:visible(video), statusLine:status?.textContent||null, duration:audio.duration, currentTime:audio.currentTime, paused:audio.paused, readyState:audio.readyState, mediaError:audio.error? audio.error.code : null, previewControlVisible:visible(previewBtn), previewControlTitle:previewBtn?.title||null}; })()' --json || fail "cannot inspect MP3 audio surface"
browser_step 15-tone-screenshot screenshot --full "$artifacts/audio-mp3.png" || fail "audio screenshot failed"

# B: Ogg Vorbis is not trusted (iOS cannot decode it), so the viewer converts
# first and plays the MP3 derivative.
browser_step 16-search-ogg fill '#filePickerInput' 'speech.ogg' || fail "cannot search speech.ogg"
browser_step 17-ogg-options eval '(() => Array.from(document.querySelectorAll("#filePickerMenu .fileMenuItem")).map((n)=>({id:n.id,text:n.innerText})))()' --json || fail "cannot inspect speech.ogg picker rows"
browser_step 18-open-ogg click '#filePickerMenu .fileMenuItem' || fail "speech.ogg picker row missing"
browser_step 19-ogg-wait wait 6000 || fail "speech.ogg conversion did not settle"
browser_step 20-ogg-surface eval '(async () => { const audio=document.querySelector("#fileAudio"); const status=document.querySelector("#fileStatus"); if (!audio) return {missing:true}; try { await audio.play(); } catch (e) { return {playError:String(e), src:audio.currentSrc||audio.getAttribute("src")}; } await new Promise((r)=>setTimeout(r,700)); return {src:audio.currentSrc||audio.getAttribute("src"), converted:(audio.currentSrc||"").includes("/file/media_preview"), statusLine:status?.textContent||null, duration:audio.duration, currentTime:audio.currentTime, paused:audio.paused, readyState:audio.readyState, mediaError:audio.error? audio.error.code : null}; })()' --json || fail "cannot inspect Ogg audio surface"
browser_step 21-ogg-derived eval '(async () => { const audio=document.querySelector("#fileAudio"); const response=await fetch(audio.currentSrc||audio.getAttribute("src"), {credentials:"same-origin", headers:{Range:"bytes=0-0"}}); return {status:response.status, contentType:response.headers.get("content-type"), contentRange:response.headers.get("content-range"), acceptRanges:response.headers.get("accept-ranges")}; })()' --json || fail "cannot inspect converted audio response"
browser_step 22-ogg-screenshot screenshot --full "$artifacts/audio-ogg-converted.png" || fail "converted audio screenshot failed"

# C: closing the viewer stops playback and drops the source.
browser_step 23-close-ogg click '#fileCloseBtn' || fail "cannot close audio viewer"
browser_step 24-close-wait wait 800 || fail "viewer close did not settle"
browser_step 25-stopped eval '(() => { const audio=document.querySelector("#fileAudio"); const visible=(n)=>!!n&&getComputedStyle(n).display!=="none"; return {src:audio?.getAttribute("src")||null, paused:audio?.paused ?? null, audioVisible:visible(audio), viewerOpen:visible(document.querySelector("#fileViewer"))}; })()' --json || fail "cannot inspect stopped audio element"

# D: video regression — MKV still converts to MP4 and plays.
browser_step 26-open-picker-video click '#fileBtn' || fail "file picker button missing"
browser_step 27-search-mkv fill '#filePickerInput' 'clip.mkv' || fail "cannot search clip.mkv"
browser_step 28-mkv-options eval '(() => Array.from(document.querySelectorAll("#filePickerMenu .fileMenuItem")).map((n)=>({id:n.id,text:n.innerText})))()' --json || fail "cannot inspect clip.mkv picker rows"
browser_step 29-open-mkv click '#filePickerMenu .fileMenuItem' || fail "clip.mkv picker row missing"
browser_step 30-mkv-wait wait 8000 || fail "clip.mkv conversion did not settle"
browser_step 31-mkv-surface eval '(async () => { const visible=(n)=>!!n&&getComputedStyle(n).display!=="none"&&n.getBoundingClientRect().width>0&&n.getBoundingClientRect().height>0; const audio=document.querySelector("#fileAudio"); const video=document.querySelector("#fileVideo"); const status=document.querySelector("#fileStatus"); if (!video) return {missing:true}; try { await video.play(); } catch (e) { return {playError:String(e), src:video.currentSrc||video.getAttribute("src")}; } await new Promise((r)=>setTimeout(r,700)); return {src:video.currentSrc||video.getAttribute("src"), converted:(video.currentSrc||"").includes("/file/media_preview"), videoVisible:visible(video), audioVisible:visible(audio), statusLine:status?.textContent||null, duration:video.duration, currentTime:video.currentTime, videoWidth:video.videoWidth, readyState:video.readyState, mediaError:video.error? video.error.code : null}; })()' --json || fail "cannot inspect MKV video surface"

browser_step 32-errors errors --json || fail "cannot collect browser errors"
browser_step 33-console console --json || fail "cannot collect browser console"
browser_step 34-final-snapshot snapshot -i -c --depth 12 || fail "cannot capture final browser snapshot"

python3 - "$artifacts" "$target_commit" <<'PY' > "$artifacts/report.json"
import json, re, sys
from pathlib import Path

artifacts = Path(sys.argv[1])
def browser_result(stem):
    path = artifacts / f"{stem}.txt"
    raw = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n{\"success\":"
    start = raw.find(marker)
    if start < 0: return {"_raw_error": raw}
    try: return json.loads(raw[start + 1:]).get("data", {}).get("result")
    except Exception as exc: return {"_raw_error": str(exc), "raw": raw}
def browser_payload(stem):
    path = artifacts / f"{stem}.txt"
    raw = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n{\"success\":"
    start = raw.find(marker)
    if start < 0: return {"_raw_error": raw}
    try: return json.loads(raw[start + 1:])
    except Exception as exc: return {"_raw_error": str(exc), "raw": raw}

tone_options = browser_result("11-tone-options")
tone = browser_result("14-tone-surface")
ogg_options = browser_result("17-ogg-options")
ogg = browser_result("20-ogg-surface")
ogg_response = browser_result("21-ogg-derived")
stopped = browser_result("25-stopped")
mkv_options = browser_result("28-mkv-options")
mkv = browser_result("31-mkv-surface")
errors = browser_payload("32-errors").get("data", {}).get("errors", [])
console = browser_payload("33-console").get("data", {}).get("messages", [])
fatal_console = [x for x in console if isinstance(x, dict) and re.search(r"\b(?:ReferenceError|TypeError|SyntaxError)\b|\bCodoxear\b.*\bfailed to load\b", str(x.get("text") or ""), re.I)]
def played(state):
    return state.get("mediaError") is None and isinstance(state.get("currentTime"), (int, float)) and state["currentTime"] > 0.1 and not state.get("paused")

checks = {
  "A_mp3_plays_from_original": bool(isinstance(tone_options, list) and tone_options and tone.get("audioVisible") is True and tone.get("videoVisible") is False and tone.get("previewSrc") is False and isinstance(tone.get("src"), str) and "/file/blob?" in tone["src"] and isinstance(tone.get("statusLine"), str) and tone["statusLine"].startswith("tone.mp3 - audio - ") and played(tone) and isinstance(tone.get("duration"), (int, float)) and 0.5 <= tone["duration"] <= 3,
  "B_ogg_converts_and_plays": bool(ogg_options and ogg.get("converted") is True and played(ogg) and isinstance(ogg.get("duration"), (int, float)) and 0.5 <= ogg["duration"] <= 3 and isinstance(ogg.get("statusLine"), str) and ogg["statusLine"].startswith("speech.ogg - compatible audio preview") and ogg_response.get("status") == 206 and ogg_response.get("contentType") == "audio/mpeg" and ogg_response.get("acceptRanges") == "bytes" and isinstance(ogg_response.get("contentRange"), str) and ogg_response["contentRange"].startswith("bytes 0-0/")),
  "C_close_stops_playback": stopped.get("src") in (None, "") and stopped.get("audioVisible") is False and stopped.get("paused") is True and stopped.get("viewerOpen") is False,
  "D_mkv_still_converts_and_plays": bool(mkv_options and isinstance(mkv.get("statusLine"), str) and mkv["statusLine"].startswith("clip.mkv - compatible video preview") and mkv.get("converted") is True and mkv.get("videoVisible") is True and mkv.get("audioVisible") is False and played(mkv) and isinstance(mkv.get("videoWidth"), int) and mkv["videoWidth"] > 0 and isinstance(mkv.get("duration"), (int, float)) and mkv["duration"] > 0.2),
  "E_no_js_errors": errors == [] and not fatal_console,
}
report = {
  "commit": sys.argv[2], "pass": all(checks.values()), "checks": checks,
  "observed": {"toneOptions":tone_options, "tone":tone, "oggOptions":ogg_options, "ogg":ogg, "oggDerived":ogg_response, "stopped":stopped, "mkvOptions":mkv_options, "mkv":mkv, "pageErrors":errors, "fatalConsole":fatal_console, "console":console},
  "artifacts": {"mp3Screenshot":str(artifacts / "audio-mp3.png"), "convertedScreenshot":str(artifacts / "audio-ogg-converted.png"), "containerLog":str(artifacts / "container.log")},
}
print(json.dumps(report, indent=2))
PY

cat "$artifacts/report.json"
if ! python3 - "$artifacts/report.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("pass") else 1)
PY
then
  echo "FAIL: audio playback verification failed; artifacts=$artifacts" >&2
  exit 1
fi
printf 'PASS: audio playback verified at %s; artifacts=%s\n' "$short_commit" "$artifacts"
