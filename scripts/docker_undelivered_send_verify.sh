#!/usr/bin/env bash
# Docker-isolated end-to-end verifier for Pi's startup deprecation gate,
# Codoxear's undelivered-send draft recovery, and the Pi extension's early
# acceptance receipt (fresh-session scenarios 5-7). The build is a Git archive
# of the commit argument; the default 8c0db06a is the historical swallow
# reproducer, so pass HEAD or a release commit to gate current code. All runtime
# state lives in this script's temporary bind-mounted container HOME, never in
# the host Codoxear runtime directory.
#
# Fresh-session scenarios: 5 is the product path (Pi renders its working
# indicator, first rollout later than the quiet window); 6 is the decisive
# control, where a test-only extension suppresses Pi's working indicator so only
# the receipt can keep the send latch from settling; 7 repeats that control with
# a prompt carrying outer whitespace and a trailing newline, which exercises
# Pi's input normalization against the receipt digest. Scenarios 5-7 fail unless
# the bridge publishes a receipt naming that prompt inside the broker's quiet
# window while the session rollout file still does not exist. For a pre-receipt
# commit set CODOXEAR_UDV_REQUIRE_RECEIPT=0: the receipt is still measured and
# reported, but only the historical assertions decide pass/fail (6-7 then act as
# negative controls and are expected to publish the false-positive verdict).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_commit="$(git -C "$repo_root" rev-parse --verify "${1:-8c0db06a}^{commit}")"
short_commit="$(git -C "$repo_root" rev-parse --short=12 "$target_commit")"
port="${CODOXEAR_UDV_PORT:-19693}"
password="docker-udv-password"
image="codoxear-udv:${short_commit}"
container="codoxear-udv-${short_commit}-${port}-$$"
root="$(mktemp -d /tmp/codoxear-udv.XXXXXX)"
artifacts="${CODOXEAR_UDV_ARTIFACTS:-$(mktemp -d /tmp/codoxear-udv-results.XXXXXX)}"
home_dir="$root/home"
readonly -a docker=(sudo docker)

[[ "$port" != "8743" ]] || { echo "refusing live port 8743" >&2; exit 2; }
[[ -z "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]] || { echo "refusing app-dir override" >&2; exit 2; }
command -v agent-browser >/dev/null || { echo "agent-browser required" >&2; exit 2; }
mkdir -p "$artifacts"
printf '%s\n' "$artifacts" > "$artifacts/artifacts-path.txt"

cleanup() {
  "${docker[@]}" logs "$container" > "$artifacts/container.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/server.log"' > "$artifacts/server.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'find "$HOME/.pi/agent/sessions" -type f -name "*.jsonl" -print -exec cat {} \;' > "$artifacts/pi-session-jsonl.txt" 2>&1 || true
  "${docker[@]}" cp "$container:/home/tester/.local/share/codoxear/send_journal" "$artifacts/send_journal" >/dev/null 2>&1 || true
  AGENT_BROWSER_SESSION="udv-$$" agent-browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  /bin/rm -rf "$root" || sudo /bin/rm -rf "$root"
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# The "slow" provider points at a tiny in-container streaming server (started by
# the Python driver for scenario 5) that delays its first response by seconds.
# The legacy file intentionally triggers Pi 0.82's unconditional interactive
# deprecation gate. There is deliberately no .agents/skills directory: this
# test is about the tools migration warning, not the project-trust dialog.
mkdir -p "$home_dir/.pi/agent/tools" "$home_dir/proj"
printf '%s\n' '// test-only legacy custom tool' > "$home_dir/.pi/agent/tools/legacy-custom-tool.js"
cat > "$home_dir/.pi/agent/models.json" <<'JSON'
{"providers":{"fake":{"baseUrl":"http://127.0.0.1:9/v1","apiKey":"fake-key","api":"openai-completions","models":[{"id":"fake-1","reasoning":false,"contextWindow":100000,"maxTokens":4096,"input":["text"]}]},"slow":{"baseUrl":"http://127.0.0.1:19694/v1","apiKey":"fake-key","api":"openai-completions","models":[{"id":"slow-1","reasoning":false,"contextWindow":100000,"maxTokens":4096,"input":["text"]}]}}}
JSON
cat > "$home_dir/.pi/agent/settings.json" <<'JSON'
{"defaultProvider":"fake","defaultModel":"fake-1","retry":{"enabled":false},"quietStartup":true}
JSON
chmod -R a+rwX "$home_dir"

# Build the exact committed application. The browser bundle is rebuilt in the
# archive context because the tracked dist bundle can lag its source modules.
ctx="$root/ctx"
mkdir -p "$ctx"
git -C "$repo_root" archive --format=tar "$target_commit" | tar -x -C "$ctx"
sed 's/git libatomic1/git libatomic1 tmux/' "$repo_root/docker/verify.Dockerfile" > "$ctx/docker/verify.Dockerfile"
npx --yes esbuild "$ctx/codoxear/static/app.js" --bundle --minify \
  --outfile="$ctx/codoxear/static/dist/app.bundle.js" --format=esm > "$artifacts/esbuild.log" 2>&1 \
  || fail "esbuild failed; see $artifacts/esbuild.log"
"${docker[@]}" build -f "$ctx/docker/verify.Dockerfile" -t "$image" "$ctx" > "$artifacts/docker-build.log" 2>&1 \
  || fail "docker build failed; see $artifacts/docker-build.log"

"${docker[@]}" run --detach --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DEFAULT_AGENT_BACKEND=pi --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 --env PI_OFFLINE=1 --env TERM=xterm-256color \
  --mount "type=bind,src=$home_dir,dst=/home/tester" --workdir /workspace \
  "$image" bash -c 'cd /workspace; python3 -m codoxear.server >"$HOME/server.log" 2>&1 & exec sleep infinity' \
  > "$artifacts/container-id.txt" || fail "container start failed"
"${docker[@]}" inspect "$container" --format '{{json .Mounts}}' > "$artifacts/container-mounts.json"

python3 "$repo_root/scripts/docker_undelivered_send_verify.py" \
  --port "$port" --password "$password" --container "$container" \
  --artifacts "$artifacts" --commit "$short_commit" --browser-session "udv-$$" \
  --mounts "$artifacts/container-mounts.json"
