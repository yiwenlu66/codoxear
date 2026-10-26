#!/usr/bin/env bash
# Docker-isolated real-browser verifier for LibreOffice-backed Office previews.
# It builds the requested committed revision from a Git archive in a disposable
# context, generates the Office fixture INSIDE the container (stdlib OOXML
# authoring + soffice legacy conversion), creates a container-only Pi session
# rooted at that fixture, and drives HTTP and a real authenticated browser
# through scripts/docker_office_preview_verify.py.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_office_preview_verify.sh [commit]

Behaviorally verifies Office previews (doc/docx/xls/xlsx/ppt/pptx) in Docker:
authenticated PDF conversion, every Excel worksheet marker (hidden sheet,
veryHidden sheet, outside-print-range cell, multi-page tail), unchanged
original downloads, source-rewrite invalidation, simultaneous conversions,
corrupt/encrypted failure paths, and the real-browser PDF.js viewer flow.
The browser phase also runs under a controlling service worker with a 6s
conversion delay, transient-failure Retry and stale-request checks, native
Office/video derivative bypass, and large-response/offline cache lifecycle
checks.
Artifacts (browser output, PDFs, extracted text, screenshots, container logs)
are retained in a fresh /tmp/codoxear-office-preview-verify.* directory.

Environment:
  CODOXEAR_OFFICE_VERIFY_PORT       Loopback port (default: 19644; never 8743)
  CODOXEAR_OFFICE_VERIFY_IMAGE      Image tag override
  CODOXEAR_OFFICE_VERIFY_PASSWORD   Login password override
  CODOXEAR_OFFICE_HELPER_CMD        In-container provisioning helper command
                                    override (default: python3 -m codoxear.dependency_doctor)
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
port="${CODOXEAR_OFFICE_VERIFY_PORT:-19644}"
password="${CODOXEAR_OFFICE_VERIFY_PASSWORD:-docker-office-preview-password}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-office-preview-verify-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-office-preview-verify.XXXXXX)"
home_dir="$root/home"
image="${CODOXEAR_OFFICE_VERIFY_IMAGE:-codoxear-office-preview-verify:${short_commit}}"
container="codoxear-office-preview-verify-${short_commit}-${port}"
browser_session="codoxear-office-preview-verify-${short_commit}-$$"
readonly repo_root target_commit short_commit port password source_pi_agent root artifacts home_dir image container browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "CODOXEAR_OFFICE_VERIFY_PORT must be a TCP port, got: $port" >&2
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

capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/container.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/server.log"' > "$artifacts/server.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\\n" "$HOME"
    printf "APP_DIR=%s\\n" "$(python3 -c "from codoxear.util import default_app_dir; print(default_app_dir())")"
    printf -- "--- fixture ---\\n"
    find /workspace/demo -maxdepth 2 -type f -printf "%y %p %s bytes\\n" 2>/dev/null || true
    printf -- "--- preview cache candidates ---\\n"
    find "$HOME/.local/share/codoxear" -type f \( -name "*.pdf" -o -name "*.mp4" \) -printf "%p %s bytes\\n" 2>/dev/null || true
    printf -- "--- socks ---\\n"
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -exec sh -c '\''echo --- "$1"; cat "$1"'\'' _ {} \; 2>/dev/null || true
  ' > "$artifacts/container-isolation.txt" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/pi-bootstrap.log" "$HOME/broker.log"' > "$artifacts/pi-and-broker.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat /workspace/demo/fixture-build.log' > "$artifacts/fixture-build.log" 2>&1 || true
  "${docker[@]}" cp "$container:/workspace/demo/manifest.json" "$artifacts/manifest.json" >/dev/null 2>&1 || true
}

cleanup() {
  capture_container_diagnostics
  AGENT_BROWSER_SESSION="$browser_session" agent-browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  sudo /bin/rm -rf "$root"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  echo "artifacts=$artifacts" >&2
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

copy_pi_config

# Build exactly the target commit. The current verifier Dockerfile is copied
# into the disposable context; no editable checkout files enter the image.
build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$build_context"
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"

# Exercise the same ESM source being committed rather than a stale tracked
# bundle; the output stays inside this disposable build context.
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"
"${docker[@]}" build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed; see $artifacts/docker-build.log"

# The fixture generator and the container bootstrap are harness tooling, not
# application source: they enter the running container via docker cp after the
# committed code has been built into the image. The container first parks on
# `sleep infinity`; the bootstrap (fixture generation, git fixture, Pi session,
# broker, server) runs as a detached exec once both files exist.
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
  "$image" sleep infinity > "$artifacts/container-id.txt" || fail "container start failed"

cat > "$root/office-bootstrap.sh" <<'BOOT'
set -eu
cd /workspace
export PYTHONPATH=/workspace
python3 /tmp/office_fixture_gen.py /workspace/demo > "$HOME/fixture-gen.json"
cd /workspace/demo
git init -q
git config user.email verify@example.test
git config user.name "Codoxear office verifier"
git add -- report.docx report.doc data.xlsx data.xls deck.pptx deck.ppt broken.docx secret.xlsx fixture-build.log manifest.json
git commit -qm fixture
cd /workspace
pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
  --session-id docker-office-preview-verify -p "Initialize isolated Office preview verification session." \
  >"$HOME/pi-bootstrap.log" 2>&1 || true
session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-office-preview-verify.jsonl" -print -quit)"
test -n "$session_log"
CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
  codoxear-broker --cwd /workspace/demo -- \
    --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
    --session "$session_log" >"$HOME/broker.log" 2>&1 &
for _ in $(seq 1 120); do
  meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
  if test -n "$meta" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"; then break; fi
  sleep 1
done
test -n "${meta:-}" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"
exec python3 -m codoxear.server >>"$HOME/server.log" 2>&1
BOOT

"${docker[@]}" cp "$repo_root/scripts/office_fixture_gen.py" "$container:/tmp/office_fixture_gen.py" \
  || fail "could not copy fixture generator into container"
"${docker[@]}" cp "$root/office-bootstrap.sh" "$container:/tmp/office-bootstrap.sh" \
  || fail "could not copy bootstrap into container"
"${docker[@]}" exec --detach "$container" bash /tmp/office-bootstrap.sh \
  || fail "container bootstrap failed to start"

wait_for_server || fail "isolated server did not become reachable"

# Execute behavior suites against the same committed source in Docker, never
# the editable checkout or live deployment. Includes prefixed SW route cases.
"${docker[@]}" exec --env PYTHONPATH=/workspace:/workspace/tests "$container" \
  python3 -m pytest -q tests/test_service_worker_offline.py tests/test_office_preview_frontend.py \
    tests/test_office_preview.py tests/test_office_file_routes.py tests/test_pdf_viewer_pipeline.py \
  > "$artifacts/focused-tests.log" 2>&1 \
  || fail "focused Docker behavior tests failed; see $artifacts/focused-tests.log"

python3 "$repo_root/scripts/docker_office_preview_verify.py" \
  --port "$port" \
  --password "$password" \
  --container "$container" \
  --artifacts "$artifacts" \
  --commit "$target_commit" \
  --browser-session "$browser_session" \
  --image "$image" \
  ${CODOXEAR_OFFICE_HELPER_CMD:+--helper-cmd "$CODOXEAR_OFFICE_HELPER_CMD"} \
  > "$artifacts/driver.log" 2>&1 || {
    tail -100 "$artifacts/driver.log" >&2 || true
    fail "Office preview verification failed; artifacts=$artifacts"
  }

tail -5 "$artifacts/driver.log" || true
printf 'PASS: Office previews verified at %s; artifacts=%s\n' "$short_commit" "$artifacts"
