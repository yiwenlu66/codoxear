#!/usr/bin/env bash
# Docker-isolated behavioral verification: the codoxear session name reaches the
# Pi session itself.
#
# Real Pi, real broker, real server, real API. No model call is made: the Pi
# session is created offline (like scripts/docker_verify.sh), which still owns a
# real session JSONL and a real bridge extension.
#
# Scenario:
#   1. a terminal-set title (`/name`) is adopted as the codoxear display name
#   2. renaming in codoxear appends exactly one `session_info` row to the Pi log
#   3. repeating the same rename appends nothing
#   4. clearing the name appends an empty `session_info` and stays cleared
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_NAME_VERIFY_PORT:-19681}"
password="${CODOXEAR_NAME_VERIFY_PASSWORD:-name-verify-password}"
commit="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
image="${CODOXEAR_NAME_VERIFY_IMAGE:-codoxear-session-name:${commit}-wt}"
container="codoxear-session-name-${port}"
root="$(mktemp -d /tmp/codoxear-session-name-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-session-name-results.XXXXXX)"
home_dir="$root/home"
readonly repo_root port password image container root artifacts home_dir

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override" >&2
  exit 2
fi

cleanup() {
  sudo docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  sudo docker logs "$container" > "$artifacts/server.log" 2>&1 || true
  sudo docker exec "$container" sh -c 'tail -n 200 "$HOME/broker.log"' > "$artifacts/broker.log" 2>&1 || true
  sudo docker exec "$container" sh -c 'for f in "$HOME"/.pi/agent/sessions/*/*.jsonl; do echo "== $f"; tail -n 40 "$f"; done' > "$artifacts/pi-sessions.log" 2>&1 || true
  sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/socks/*.json' > "$artifacts/sidecars.json" 2>&1 || true
  echo "artifacts=$artifacts" >&2
  exit 1
}

# --- working-tree build context (same construction as the other verifiers) ---
build_context="$root/build-context"
mkdir -p "$build_context/docker"
git -C "$repo_root" ls-files -z --cached --others --exclude-standard | python3 -c '
import sys
skip = (b".memory/", b"review/")
parts = sys.stdin.buffer.read().split(b"\0")
keep = [p for p in parts if p and not p.startswith(skip)]
sys.stdout.buffer.write(b"\0".join(keep) + b"\0")
' > "$root/context-files.nul"
tar --null --no-recursion -C "$repo_root" -T "$root/context-files.nul" -cf "$root/context.tar"
tar -C "$build_context" -xf "$root/context.tar"
cp "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"

sudo docker build --file "$build_context/docker/verify.Dockerfile" --tag "$image" "$build_context" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed"
sudo docker rm -f "$container" >/dev/null 2>&1 || true

# Pi config comes from the host exactly as scripts/docker_verify.sh does it: the
# verification session never needs a model reply, but Pi refuses to start a
# session without a resolvable provider.
copy_pi_config() {
  umask 077
  mkdir -p "$home_dir/.pi/agent" "$home_dir/.local/share"
  local name
  for name in models.json models-store.json auth.json settings.json trust.json anthropic-context-management.json; do
    if [[ -r "$HOME/.pi/agent/$name" ]]; then
      install -m 600 "$HOME/.pi/agent/$name" "$home_dir/.pi/agent/$name"
    fi
  done
}
copy_pi_config
sudo chown -R 1000:1000 "$home_dir"
sudo docker run --detach \
  --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.1 \
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
      --session-id name-verify-session -p "Initialize isolated session-name verification." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*name-verify-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$session_log" >"$HOME/broker.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"; then
        break
      fi
      sleep 1
    done
    exec python3 -m codoxear.server
  ' > "$artifacts/container-id.txt" || fail "container start failed"

for _ in $(seq 1 200); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
  [[ "$code" == "401" ]] && break
  sleep 0.25
done
[[ "${code:-}" == "401" ]] || fail "server readiness failed"

cookie="$root/cookies.txt"
curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
  --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/api-login.json"

session_id=""
for _ in $(seq 1 240); do
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions.json" || true
  session_id="$(python3 - "$artifacts/sessions.json" <<'PY'
import json, sys
try:
    rows = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if isinstance(rows, dict):
    rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and row.get("session_id"):
        print(row["session_id"])
        break
PY
)"
  [[ -n "$session_id" ]] && break
  sleep 0.25
done
[[ -n "$session_id" ]] || fail "Pi session never appeared in /api/sessions"

log_path="$(python3 - "$artifacts/sessions.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(rows, dict):
    rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi":
        print(row.get("log_path") or "")
        break
PY
)"
[[ -n "$log_path" ]] || fail "Pi session has no bound log path"

log_rows() {
  sudo docker exec -i "$container" python3 - "$log_path" <<'PY'
import json, sys
count = 0
names = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("type") == "session_info":
            count += 1
            names.append(obj.get("name"))
print(json.dumps({"count": count, "names": names}))
PY
}

row_field() {
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" \
    | python3 -c '
import json, sys
rows = json.load(sys.stdin)
if isinstance(rows, dict):
    rows = rows.get("sessions") or []
sid = sys.argv[1]
field = sys.argv[2]
for row in rows:
    if isinstance(row, dict) and row.get("session_id") == sid:
        print(row.get(field) if row.get(field) is not None else "")
        break
' "$session_id" "$1"
}

rename() {
  curl -fsS -b "$cookie" -H 'Content-Type: application/json' \
    --data "{\"name\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" \
    "http://127.0.0.1:${port}/api/sessions/${session_id}/rename" > "$artifacts/rename-${2}.json"
}

# --- Step 0: the live bridge advertises the capability -----------------------
for _ in $(seq 1 120); do
  control="$(row_field session_name_control || true)"
  [[ "$control" == "True" ]] && break
  sleep 0.25
done
[[ "${control:-}" == "True" ]] || fail "bridge never advertised session_name_control (got ${control:-unset})"

# --- Step 1: a title typed in the terminal is adopted ------------------------
curl -fsS -b "$cookie" -H 'Content-Type: application/json' \
  --data '{"text":"/name terminal title"}' \
  "http://127.0.0.1:${port}/api/sessions/${session_id}/send" > "$artifacts/send-name.json" || fail "send of /name failed"

for _ in $(seq 1 120); do
  rows="$(log_rows)"
  alias_value="$(row_field alias || true)"
  if python3 - "$rows" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
raise SystemExit(0 if "terminal title" in (data.get("names") or []) else 1)
PY
  then
    break
  fi
  sleep 0.25
done
python3 - "$rows" "$artifacts/adopt.json" <<'PY' || fail "the Pi-side /name never reached the session log"
import json, sys
data = json.loads(sys.argv[1])
assert "terminal title" in (data.get("names") or []), data
json.dump(data, open(sys.argv[2], "w", encoding="utf-8"))
PY

for _ in $(seq 1 120); do
  alias_value="$(row_field alias || true)"
  [[ "$alias_value" == "terminal title" ]] && break
  sleep 0.25
done
[[ "${alias_value:-}" == "terminal title" ]] || fail "terminal title was not adopted as the display name (got ${alias_value:-unset})"

# --- Step 2: a codoxear rename reaches Pi -----------------------------------
rename "web name" 1
python3 - "$artifacts/rename-1.json" <<'PY' || fail "rename response did not report a projection"
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
assert body.get("ok") is True, body
assert body.get("alias") == "web name", body
prop = body.get("name_propagation") or {}
assert prop.get("ok") is True, prop
assert prop.get("reason") == "applied", prop
assert prop.get("name") == "web name", prop
PY

for _ in $(seq 1 120); do
  rows="$(log_rows)"
  if python3 - "$rows" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
raise SystemExit(0 if (data.get("names") or [])[-1:] == ["web name"] else 1)
PY
  then
    break
  fi
  sleep 0.25
done
python3 - "$rows" "$artifacts/after-rename.json" <<'PY' || fail "the rename never reached the Pi session log"
import json, sys
data = json.loads(sys.argv[1])
assert (data.get("names") or [])[-1:] == ["web name"], data
assert data.get("count") == 2, data
json.dump(data, open(sys.argv[2], "w", encoding="utf-8"))
PY

# --- Step 3: repeating the same title is a no-op ----------------------------
rename "web name" 2
python3 - "$artifacts/rename-2.json" <<'PY' || fail "repeated rename was not reported as unchanged"
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
prop = body.get("name_propagation") or {}
assert prop.get("reason") == "unchanged", prop
PY
sleep 1.5
rows="$(log_rows)"
python3 - "$rows" <<'PY' || fail "repeated rename appended a duplicate session_info row"
import json, sys
data = json.loads(sys.argv[1])
assert data.get("count") == 2, data
PY

# --- Step 4: clearing the name clears the Pi title and stays cleared --------
rename "" 3
for _ in $(seq 1 120); do
  rows="$(log_rows)"
  if python3 - "$rows" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
raise SystemExit(0 if (data.get("names") or [])[-1:] == [""] else 1)
PY
  then
    break
  fi
  sleep 0.25
done
python3 - "$rows" <<'PY' || fail "clearing the alias did not clear the Pi session title"
import json, sys
data = json.loads(sys.argv[1])
assert (data.get("names") or [])[-1:] == [""], data
assert data.get("count") == 3, data
PY

# The cleared identity stays cleared: the previous title is not re-adopted.
sleep 2.5
alias_value="$(row_field alias || true)"
[[ -z "${alias_value:-}" ]] || fail "cleared name was re-adopted (alias=${alias_value})"

echo "PASS: Pi session name propagation verified (session ${session_id})"
echo "artifacts=$artifacts"
