#!/usr/bin/env bash
# Docker-isolated behavioral verifier for stale-cache transport visibility.
#
# Scenario: the app is loaded while the server is healthy (service worker
# activates and warms its offline API caches), then the container is paused —
# the server stops answering without the client ever going "offline". Before
# the fix this rendered cached content silently forever; now the network
# banner must appear with the "showing cached data" variant within a poll
# cycle, and disappear again after the container is unpaused.
#
# Same isolation contract as scripts/docker_verify.sh: the app is built from
# a Git archive of the requested commit, runs under a container-only HOME,
# and is driven through the host's agent-browser on a loopback port that is
# never the live 8743.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_stale_banner_verify.sh [commit]

Environment:
  CODOXEAR_STALE_VERIFY_PORT       Loopback port (default: 19671; never 8743)
  CODOXEAR_STALE_VERIFY_IMAGE      Image tag (default: codoxear-verify:<commit>)
  CODOXEAR_STALE_VERIFY_PASSWORD   Login password (default: docker-stale-password)
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
port="${CODOXEAR_STALE_VERIFY_PORT:-19671}"
password="${CODOXEAR_STALE_VERIFY_PASSWORD:-docker-stale-password}"
image="${CODOXEAR_STALE_VERIFY_IMAGE:-codoxear-verify:${short_commit}}"
container="codoxear-stale-verify-${short_commit}-${port}-$$"
root="$(mktemp -d /tmp/codoxear-stale-verify-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-stale-verify.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-stale-verify-${short_commit}-$$"
readonly root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
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
  "${docker[@]}" unpause "$container" >/dev/null 2>&1 || true
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
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
  const b = document.getElementById("networkBanner");
  return {
    bannerText: b ? b.textContent : null,
    bannerHidden: b ? b.hidden : null,
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
# Always build: the bundle rebuild above changes the context even for a
# previously built commit, and layer caching keeps a no-change build cheap.
"${docker[@]}" build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed; see $artifacts/docker-build.log"
printf 'built image %s\n' "$image" > "$artifacts/docker-image.txt"

"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach \
  --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" \
  --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" > "$artifacts/container-id.txt" || fail "container start failed"

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
[[ "$(probe_field 01-baseline bannerHidden)" == "True" ]] || fail "banner visible while server healthy"

# Freeze the server: requests hang, the worker's network-first race times out,
# and every API poll is answered from the offline cache.
"${docker[@]}" pause "$container" > /dev/null
browser wait 16000 > /dev/null 2>&1
probe 02-server-paused
[[ "$(probe_field 02-server-paused onLine)" == "True" ]] || fail "test premise broken: navigator went offline"
[[ "$(probe_field 02-server-paused bannerHidden)" == "False" ]] || fail "banner stayed hidden while server unreachable (stale content rendered as fresh)"
banner_text="$(probe_field 02-server-paused bannerText)"
case "$banner_text" in
  *"cached data"*) ;;
  *) fail "banner text did not report cached data, got: $banner_text" ;;
esac
browser screenshot "$artifacts/02-server-paused.png" > /dev/null 2>&1 || true

# Recovery: after unpause the suspect-window revalidation clears, a live poll
# succeeds, and the banner hides again.
"${docker[@]}" unpause "$container" > /dev/null
browser wait 22000 > /dev/null 2>&1
probe 03-server-recovered
[[ "$(probe_field 03-server-recovered bannerHidden)" == "True" ]] || fail "banner stayed visible after server recovery"
browser screenshot "$artifacts/03-server-recovered.png" > /dev/null 2>&1 || true

printf 'PASS: stale-cache banner appears on server outage and clears on recovery. Artifacts: %s\n' "$artifacts"
