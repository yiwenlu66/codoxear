#!/usr/bin/env bash
# Docker-isolated repro: a session that already shows the app's default name, then
# gets a user-chosen name, flips back to the default a few seconds later.
#
# Real Pi, real broker, real server, real API. The broker is pre-created (as in
# scripts/docker_session_name_stall_verify.sh) so the Pi bridge is live and the
# app's default name actually reaches Pi -- the precondition every live Pi log
# shows (the cwd-basename session_info row that precedes the chosen name).
#
# Sequence:
#   0. view the session -> the app assigns its default (cwd basename) and pushes it
#   1. rename the session in codoxear -> reported applied
#   2. keep polling -> the alias reverts to the default
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_FLIP_VERIFY_PORT:-19693}"
wanted="My Required Name"
password="${CODOXEAR_FLIP_VERIFY_PASSWORD:-flip-verify-password}"
commit="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
image="${CODOXEAR_FLIP_VERIFY_IMAGE:-codoxear-name-flip:${commit}-wt}"
container="codoxear-name-flip-${port}"
root="$(mktemp -d /tmp/codoxear-name-flip-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-name-flip-results.XXXXXX)"
home_dir="$root/home"
readonly repo_root port password wanted image container root artifacts home_dir

if [[ "$port" == "8743" ]]; then echo "refusing live Codoxear port 8743" >&2; exit 2; fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then echo "refusing host app-dir override" >&2; exit 2; fi
cleanup() { sudo docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
dump_diagnostics() {
  sudo docker logs "$container" > "$artifacts/server.log" 2>&1 || true
  sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/session_aliases.json' > "$artifacts/aliases.json" 2>/dev/null || true
  sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/session_names.json' > "$artifacts/names.json" 2>/dev/null || true
  sudo docker exec "$container" sh -c 'tail -n 80 "$HOME/broker.log"' > "$artifacts/broker.log" 2>/dev/null || true
}
fail() { echo "FAIL: $*" >&2; dump_diagnostics; echo "artifacts=$artifacts" >&2; exit 1; }

build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" ls-files -z --cached --others --exclude-standard | python3 -c '
import sys
skip = (b".memory/", b"review/")
parts = sys.stdin.buffer.read().split(b"\0")
keep = [p for p in parts if p and not p.startswith(skip)]
sys.stdout.buffer.write(b"\0".join(keep) + b"\0")' > "$root/context-files.nul"
tar --null --no-recursion -C "$repo_root" -T "$root/context-files.nul" -cf "$root/context.tar"
tar -C "$build_context" -xf "$root/context.tar"
cp "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"
sudo docker build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed"
sudo docker rm -f "$container" >/dev/null 2>&1 || true

umask 077
mkdir -p "$home_dir/.pi/agent" "$home_dir/.local/share"
for name in models.json models-store.json auth.json settings.json trust.json anthropic-context-management.json; do
  if [[ -r "$HOME/.pi/agent/$name" ]]; then install -m 600 "$HOME/.pi/agent/$name" "$home_dir/.pi/agent/$name"; fi
done
sudo chown -R 1000:1000 "$home_dir"
sudo docker run --detach --name "$container" --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester --env PI_OFFLINE=1 --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.1 --env CODEX_WEB_DEFAULT_AGENT_BACKEND=pi \
  --env PYTHONPATH=/workspace --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$home_dir,dst=/home/tester" --workdir /workspace "$image" \
  bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id name-flip-session -p "Initialize isolated session-name flip repro." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*name-flip-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$session_log" >"$HOME/broker.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\"session_id\"[[:space:]]*:[[:space:]]*\"[^\"]+\"" "$meta"; then break; fi
      sleep 1
    done
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

for _ in $(seq 1 200); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
  [[ "$code" == "401" ]] && break; sleep 0.25
done
[[ "${code:-}" == "401" ]] || fail "server readiness failed"
cookie="$root/cookies.txt"
curl -fsS -c "$cookie" -H 'Content-Type: application/json' --data "{\"password\":\"${password}\"}" \
  "http://127.0.0.1:${port}/api/login" > "$artifacts/api-login.json"

session_id=""
for _ in $(seq 1 240); do
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/latest-sessions.json" || true
  session_id="$(python3 - "$artifacts/latest-sessions.json" <<'PY'
import json, sys
try: rows = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception: raise SystemExit(0)
if isinstance(rows, dict): rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and row.get("session_id"):
        print(row["session_id"]); break
PY
)"
  [[ -n "$session_id" ]] && break; sleep 0.25
done
[[ -n "$session_id" ]] || fail "Pi session never appeared"

log_path="$(python3 - "$artifacts/latest-sessions.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(rows, dict): rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi":
        print(row.get("log_path") or ""); break
PY
)"
row_val() {
  python3 - "$artifacts/latest-sessions.json" "$session_id" "$1" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(rows, dict): rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and str(row.get("session_id")) == sys.argv[2]:
        v = row.get(sys.argv[3]); print("" if v is None else v); break
PY
}
pi_names() {
  [[ -n "$log_path" ]] || { echo '[]'; return; }
  sudo docker exec -i "$container" python3 - "$log_path" <<'PY'
import json, sys
names = []
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except ValueError: continue
            if obj.get("type") == "session_info": names.append(obj.get("name"))
except FileNotFoundError:
    pass
print(json.dumps(names))
PY
}
refresh() {
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/latest-sessions.json" 2>/dev/null || true
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions/${session_id}/messages/export" >/dev/null 2>&1 || true
}

# Step 0: the app assigns its default and it reaches Pi (observed title == default).
default_name=""
for _ in $(seq 1 120); do
  refresh
  observed="$(row_val backend_session_name)"
  alias_value="$(row_val alias)"
  if [[ -n "$alias_value" && "$alias_value" == "$observed" ]]; then default_name="$alias_value"; break; fi
  sleep 0.25
done
[[ -n "$default_name" ]] || fail "the app's default never reached Pi (alias=$(row_val alias), observed=$(row_val backend_session_name))"
echo "STEP 0: app assigned default '${default_name}'; Pi title is now $(pi_names)"

# Step 1: rename to the chosen name.
curl -fsS -b "$cookie" -H 'Content-Type: application/json' \
  --data "{\"name\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$wanted")}" \
  "http://127.0.0.1:${port}/api/sessions/${session_id}/rename" > "$artifacts/rename.json" || fail "rename failed"
python3 - "$artifacts/rename.json" <<'PY' || fail "rename was not applied"
import json, sys
b = json.load(open(sys.argv[1], encoding="utf-8"))
assert b.get("alias") == "My Required Name", b
PY
echo "STEP 1: renamed to '${wanted}' (response alias=$(python3 -c 'import json;print(json.load(open("'"$artifacts/rename.json"'"))["alias"])'))"

# Step 2: keep polling; the alias flips back.
echo "t(s)  alias                 pi-title"
flipped=0
for i in $(seq 0 30); do
  refresh
  alias_value="$(row_val alias)"
  printf '%4s  %-20s  %s\n' "$i" "${alias_value:-(empty)}" "$(pi_names)"
  if [[ "$alias_value" == "$default_name" ]]; then flipped=1; fi
  sleep 0.5
done

dump_diagnostics
echo "--- final name projection ---"; cat "$artifacts/names.json"
if [[ "$flipped" == "1" ]]; then
  echo "REPRODUCED: chosen '${wanted}' flipped back to the default '${default_name}'"
else
  echo "PASS (no flip): the chosen name survived"
fi
echo "artifacts=$artifacts"
