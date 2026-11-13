#!/usr/bin/env bash
# Real-browser geometry verification for the in-flight attachment chip.
#
# The plain "uploading" chip must render the same single-line pill as the
# completed chip; only the failure/recovery layout may grow. The upload is held
# open by the Docker-only fault proxy so the in-flight chip can be measured with
# getBoundingClientRect/elementFromPoint before its commit replaces it.
#
# Source modes:
#   (default)     committed HEAD source plus the working tree's app.css overlay,
#                 so an uncommitted stylesheet fix is what renders while every
#                 other file stays at HEAD (a concurrent editor's half-applied
#                 change can never leak into this measurement)
#   --baseline    committed HEAD source only (reproduces the pre-fix mismatch)
#   --css <path>  overlay a different app.css
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"

mode="fix"
ref="HEAD"
css="$repo/codoxear/static/app.css"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) mode="baseline"; shift ;;
    --fix) mode="fix"; shift ;;
    --commit) ref="${2:?--commit needs a ref}"; shift 2 ;;
    --css) css="${2:?--css needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

expect="fixed"
if [[ "$mode" == "baseline" ]]; then expect="baseline"; fi

port="${CODOXEAR_PILL_VERIFY_PORT:-19893}"
[[ "$port" != 8743 ]] || exit 2
image="${CODOXEAR_PILL_VERIFY_IMAGE:-codoxear-iso3:latest}"
root="$(mktemp -d /tmp/codoxear-pill-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-pill-results.XXXXXX)"
container="codoxear-pill-verify-${port}-$$"
session="pill-verify-$$"
cleanup() {
  docker logs "$container" > "$artifacts/server.log" 2>&1 || true
  docker cp "$container:/tmp/proxy.log" "$artifacts/proxy.log" >/dev/null 2>&1 || true
  docker cp "$container:/tmp/broker.log" "$artifacts/broker.log" >/dev/null 2>&1 || true
  echo "artifacts=$artifacts"
  agent-browser --session "$session" close >/dev/null 2>&1 || true
  agent-browser --session "${session}-coarse" close >/dev/null 2>&1 || true
  [[ -n "${CODOXEAR_PILL_VERIFY_KEEP:-}" ]] || docker rm -f "$container" >/dev/null 2>&1 || true
  /bin/rm -rf "$root"
}
trap cleanup EXIT

mkdir -p "$root/source"
commit="$(git -C "$repo" rev-parse "${ref}^{commit}")"
git -C "$repo" archive "$commit" | tar -x -C "$root/source"
if [[ "$mode" == "fix" ]]; then
  [[ -f "$css" ]] || { echo "missing stylesheet overlay: $css" >&2; exit 1; }
  cp "$css" "$root/source/codoxear/static/app.css"
  printf '%s\n' "$css" > "$artifacts/css-overlay.txt"
fi
printf '%s\n' "$commit" > "$artifacts/source.txt"
# Harness files come from the working tree so a committed snapshot can be
# measured with the current fault proxy and fixture broker.
cp "$repo/tests/fixtures/upload_broker.py" "$repo/tests/fixtures/upload_fault_proxy.py" "$root/source/tests/fixtures/"

docker run -d --name "$container" -p "127.0.0.1:${port}:19881" \
  -e HOME=/tmp/pill-home -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  -e CODEX_WEB_PASSWORD=pill-verify-password -e CODEX_WEB_HOST=127.0.0.1 -e CODEX_WEB_PORT=19882 \
  -e CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.1 \
  -v "$root/source:/workspace:ro" -w /workspace "$image" sh -c '
    mkdir -p "$HOME"
    python3 tests/fixtures/upload_broker.py > /tmp/broker.log 2>&1 &
    python3 tests/fixtures/upload_fault_proxy.py > /tmp/proxy.log 2>&1 &
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt"
for _ in $(seq 1 80); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" || true)"
  [[ "$code" == "401" ]] && break
  sleep .25
done
code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" || true)"
if [[ "$code" != "401" ]]; then
  echo "server never became ready on port ${port} (last /api/me -> ${code})" >&2
  exit 1
fi
python3 "$repo/scripts/docker_upload_pill_geometry_verify.py" --port "$port" --artifacts "$artifacts" \
  --session "$session" --expect "$expect" || {
  agent-browser --session "$session" screenshot "$artifacts/failure.png" >/dev/null 2>&1 || true
  exit 1
}
echo "artifacts=$artifacts"
