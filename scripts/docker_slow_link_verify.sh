#!/usr/bin/env bash
# Docker-isolated behavioral verifier for slow-link discrimination.
#
# Scenario: the app is loaded through a loopback delay proxy with zero delay
# (service worker activates and warms its caches), then the proxy starts
# adding 5 seconds to every response. The connection is slow, not dead: every
# fetch eventually succeeds. The connection icon must stay HIDDEN the whole
# time — before the detection fix, the first 3s race loss armed the suspect
# latch and every cached answer read as "Connection unavailable".
#
# Same isolation contract as scripts/docker_verify.sh: the app is built from
# a Git archive of the requested commit, runs under a container-only HOME,
# and is driven through the host's agent-browser on loopback ports that never
# include the live 8743.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_slow_link_verify.sh [commit]

Environment:
  CODOXEAR_SLOW_VERIFY_PORT          Browser-facing proxy port (default: 19672; never 8743)
  CODOXEAR_SLOW_VERIFY_UPSTREAM_PORT Container server port (default: 19673; never 8743)
  CODOXEAR_SLOW_VERIFY_IMAGE         Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_SLOW_VERIFY_PASSWORD      Login password (default: docker-slow-password)
  CODOXEAR_SLOW_LINK_DELAY_SECONDS   Injected per-response delay (default: 5)
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
port="${CODOXEAR_SLOW_VERIFY_PORT:-19672}"
upstream_port="${CODOXEAR_SLOW_VERIFY_UPSTREAM_PORT:-19673}"
password="${CODOXEAR_SLOW_VERIFY_PASSWORD:-docker-slow-password}"
delay_seconds="${CODOXEAR_SLOW_LINK_DELAY_SECONDS:-5}"
image="${CODOXEAR_SLOW_VERIFY_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-slow-verify-${short_commit}-${port}-$$"
root="$(mktemp -d /tmp/codoxear-slow-verify-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-slow-verify.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-slow-verify-${short_commit}-$$"
delay_file="$root/delay_seconds"
proxy_pid=""
readonly root artifacts home_dir browser_session delay_file
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" || "$upstream_port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override; unset CODEXEAR_APP_DIR and CODEX_WEB_APP_DIR" >&2
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
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  if [[ -n "$proxy_pid" ]]; then kill "$proxy_pid" >/dev/null 2>&1 || true; fi
  /bin/rm -rf "$root" || sudo /bin/rm -rf "$root"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
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

PROBE='(() => {
  const s = document.getElementById("networkStatus");
  const spoken = s ? s.querySelector(".sr-only") : null;
  return {
    statusSpoken: spoken ? spoken.textContent : null,
    statusHidden: s ? s.hidden : null,
    swControlled: Boolean(navigator.serviceWorker && navigator.serviceWorker.controller),
    appRendered: Boolean(document.querySelector(".app .topbar")),
    onLine: navigator.onLine,
  };
})()'

probe() {
  browser eval "$PROBE" --json > "$artifacts/$1.json" 2>&1 || fail "probe $1 failed"
  cat "$artifacts/$1.json"
  echo
}

probe_field() {
  python3 -c 'import json,sys
data = json.load(open(sys.argv[1]))
while isinstance(data, dict) and "result" not in data and "data" in data:
    data = data["data"]
if isinstance(data, dict) and "result" in data:
    data = data["result"]
print(data.get(sys.argv[2]))' "$artifacts/$1.json" "$2"
}

umask 077
mkdir -p "$home_dir/.local/share"
echo 0 > "$delay_file"

build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"

# Rebuild the browser bundle from the archived ESM source with the same esbuild
# invocation deploy.sh uses; the tracked bundle in the archive may predate the
# commit under test.
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"
"${docker[@]}" build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed; see $artifacts/docker-build.log"
printf 'built image %s\n' "$image" > "$artifacts/docker-image.txt"

"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach \
  --name "$container" \
  --publish "127.0.0.1:${upstream_port}:${upstream_port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$upstream_port" \
  --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" > "$artifacts/container-id.txt" || fail "container start failed"

python3 "$repo_root/scripts/delay_proxy.py" "$port" "$upstream_port" "$delay_file" \
  > "$artifacts/proxy.log" 2>&1 &
proxy_pid="$!"

wait_for_server || fail "server readiness check failed"

browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser could not open app"
browser fill '#pw' "$password" > /dev/null 2>&1 || fail "login password field was not available"
browser click '#loginBtn' > /dev/null 2>&1 || fail "login button was not available"

# Wait for the service worker to activate, claim the page, and warm its API
# caches, then reload so the page is SW-controlled from the first poll.
controlled=""
for _ in $(seq 1 30); do
  probe 00-sw-wait > /dev/null
  controlled="$(probe_field 00-sw-wait swControlled)"
  [[ "$controlled" == "True" || "$controlled" == "true" ]] && break
  sleep 1
done
[[ "$controlled" == "True" || "$controlled" == "true" ]] || fail "service worker never controlled the page"
browser open "http://127.0.0.1:${port}/" > /dev/null 2>&1 || fail "browser reload failed"
browser wait 6000 > /dev/null 2>&1

probe 01-baseline
[[ "$(probe_field 01-baseline appRendered)" == "True" ]] || fail "app did not render while online"
[[ "$(probe_field 01-baseline statusHidden)" == "True" ]] || fail "connection icon visible while server healthy"

# Slow the link: every response now takes ${delay_seconds}s but always succeeds.
# Across several poll cycles the connection icon must never appear.
echo "$delay_seconds" > "$delay_file"
for round in 1 2 3 4; do
  browser wait 8000 > /dev/null 2>&1
  probe "02-slow-${round}"
  [[ "$(probe_field 02-slow-${round} onLine)" == "True" ]] || fail "test premise broken: navigator went offline"
  [[ "$(probe_field 02-slow-${round} appRendered)" == "True" ]] || fail "app stopped rendering on a slow link"
  [[ "$(probe_field 02-slow-${round} statusHidden)" == "True" ]] \
    || fail "connection icon appeared on a slow-but-working link: $(probe_field 02-slow-${round} statusSpoken)"
done
browser screenshot "$artifacts/02-slow-link.png" > /dev/null 2>&1 || true

# Recovery: delay back to zero, polls return to fast live answers.
echo 0 > "$delay_file"
browser wait 8000 > /dev/null 2>&1
probe 03-recovered
[[ "$(probe_field 03-recovered statusHidden)" == "True" ]] || fail "connection icon visible after the link recovered"

printf 'PASS: connection icon stayed hidden on a slow-but-working link. Artifacts: %s\n' "$artifacts"
