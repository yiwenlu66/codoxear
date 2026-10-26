#!/usr/bin/env bash
# Committed application snapshot, fake read-only broker, real browser and HTTP
# proxy faults. No provider config, host HOME sandbox, or production requests.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
commit="$(git -C "$repo" rev-parse "${1:-HEAD}^{commit}")"
port="${CODOXEAR_UPLOAD_VERIFY_PORT:-19881}"
[[ "$port" != 8743 ]] || exit 2
image="${CODOXEAR_UPLOAD_VERIFY_IMAGE:-codoxear-iso3:latest}"
root="$(mktemp -d /tmp/codoxear-upload-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-upload-results.XXXXXX)"
container="codoxear-upload-verify-${port}-$$"
session="upload-verify-$$"
cleanup() {
  docker logs "$container" > "$artifacts/server.log" 2>&1 || true
  docker cp "$container:/tmp/proxy.log" "$artifacts/proxy.log" >/dev/null 2>&1 || true
  docker cp "$container:/tmp/broker.log" "$artifacts/broker.log" >/dev/null 2>&1 || true
  echo "artifacts=$artifacts"
  agent-browser --session "$session" close >/dev/null 2>&1 || true
  docker rm -f "$container" >/dev/null 2>&1 || true
  /bin/rm -rf "$root"
}
trap cleanup EXIT
mkdir -p "$root/source"
git -C "$repo" archive "$commit" | tar -x -C "$root/source"
printf '%s\n' "$commit" > "$artifacts/commit.txt"
# Harness files are copied explicitly so it can verify a preceding checkpoint.
mkdir -p "$root/source/tests/fixtures"
cp "$repo/tests/fixtures/upload_broker.py" "$repo/tests/fixtures/upload_fault_proxy.py" "$root/source/tests/fixtures/"
docker run -d --name "$container" -p "127.0.0.1:${port}:19881" \
  -e HOME=/tmp/upload-home -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  -e CODEX_WEB_PASSWORD=upload-verify-password -e CODEX_WEB_HOST=127.0.0.1 -e CODEX_WEB_PORT=19882 \
  -e CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.1 \
  -v "$root/source:/workspace:ro" -w /workspace "$image" sh -c '
    mkdir -p "$HOME"
    python3 tests/fixtures/upload_broker.py > /tmp/broker.log 2>&1 &
    python3 tests/fixtures/upload_fault_proxy.py > /tmp/proxy.log 2>&1 &
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt"
for _ in $(seq 1 80); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" || true)" == 401 ]] && break
  sleep .25
done
python3 "$repo/scripts/docker_upload_robustness_verify.py" --port "$port" --container "$container" --artifacts "$artifacts" --session "$session" || {
  agent-browser --session "$session" screenshot "$artifacts/failure.png" >/dev/null 2>&1 || true
  exit 1
}
echo "artifacts=$artifacts"
