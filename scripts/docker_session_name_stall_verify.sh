#!/usr/bin/env bash
# Docker-isolated behavioral repro: a rename issued while the AI agent process is
# not servicing its event loop fails, even though the rename itself succeeded.
#
# Hypothesis under test (from the live incident on 2026-11-22):
#   codoxear's session name is pushed into the running agent over a Unix socket
#   that the *agent's own event loop* services, synchronously, on the HTTP
#   request path, with a 1.5 s bridge budget inside a 2.0 s broker budget. When
#   the agent is busy (its event loop blocked -- which is exactly what happens
#   while it is booting or working), the write times out; the timeout escapes the
#   write helper as a generic error; the name projection records a failure; and
#   the rename API still reports the alias as changed. The user is told the
#   rename failed while the agent keeps whatever title it had.
#
# This script isolates the mechanism by SIGSTOPping the real agent process: the
# kernel keeps the listening socket accepting, but the process cannot read or
# reply, exactly as a blocked event loop behaves from the caller's side.
#
#   1. healthy agent: a rename propagates normally (baseline)
#   2. agent frozen: the rename is recorded but reported as failed, and the
#      request blocks for the bridge timeout
#   3. agent unfrozen: the name it was told had failed appears anyway, because
#      the request had already been delivered to the agent's socket queue
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_STALL_VERIFY_PORT:-19691}"
password="${CODOXEAR_STALL_VERIFY_PASSWORD:-stall-verify-password}"
commit="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
image="${CODOXEAR_STALL_VERIFY_IMAGE:-codoxear-name-stall:${commit}-wt}"
container="codoxear-name-stall-${port}"
root="$(mktemp -d /tmp/codoxear-name-stall-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-name-stall-results.XXXXXX)"
home_dir="$root/home"
readonly repo_root port password image container root artifacts home_dir

if [[ "$port" == "8743" ]]; then echo "refusing live Codoxear port 8743" >&2; exit 2; fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then echo "refusing host app-dir override" >&2; exit 2; fi

cleanup() { sudo docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  sudo docker logs "$container" > "$artifacts/server.log" 2>&1 || true
  sudo docker exec "$container" sh -c 'tail -n 200 "$HOME/broker.log"' > "$artifacts/broker.log" 2>&1 || true
  sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/session_names.json 2>/dev/null' > "$artifacts/session_names.json" 2>&1 || true
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

copy_pi_config() {
  umask 077
  mkdir -p "$home_dir/.pi/agent" "$home_dir/.local/share"
  local name
  for name in models.json models-store.json auth.json settings.json trust.json anthropic-context-management.json; do
    if [[ -r "$HOME/.pi/agent/$name" ]]; then install -m 600 "$HOME/.pi/agent/$name" "$home_dir/.pi/agent/$name"; fi
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
      --session-id name-stall-session -p "Initialize isolated session-name stall repro." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*name-stall-session.jsonl" -print -quit)"
    test -n "$session_log"
    CODEX_WEB_AGENT_BACKEND=pi CODEX_WEB_OWNER=web PI_BIN=pi PI_OFFLINE=1 \
      codoxear-broker --cwd /workspace -- \
        --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
        --session "$session_log" >"$HOME/broker.log" 2>&1 &
    for _ in $(seq 1 120); do
      meta="$(find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -print -quit 2>/dev/null || true)"
      if test -n "$meta" && grep -Eq "\\\"session_id\\\"[[:space:]]*:[[:space:]]*\\\"[^\\\"]+\\\"" "$meta"; then break; fi
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
try: rows = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception: raise SystemExit(0)
if isinstance(rows, dict): rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and row.get("session_id"):
        print(row["session_id"]); break
PY
)"
  [[ -n "$session_id" ]] && break
  sleep 0.25
done
[[ -n "$session_id" ]] || fail "Pi session never appeared in /api/sessions"

log_path="$(python3 - "$artifacts/sessions.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(rows, dict): rows = rows.get("sessions") or []
for row in rows or []:
    if isinstance(row, dict) and row.get("agent_backend") == "pi":
        print(row.get("log_path") or ""); break
PY
)"
[[ -n "$log_path" ]] || fail "Pi session has no bound log path"

log_names() {
  sudo docker exec -i "$container" python3 - "$log_path" <<'PY'
import json, sys
names = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line: continue
        try: obj = json.loads(line)
        except ValueError: continue
        if obj.get("type") == "session_info": names.append(obj.get("name"))
print(json.dumps(names))
PY
}

rename() {
  curl -fsS -b "$cookie" -H 'Content-Type: application/json' \
    --data "{\"name\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" \
    "http://127.0.0.1:${port}/api/sessions/${session_id}/rename" > "$artifacts/rename-${2}.json"
}

pi_pid() {
  sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/socks/*.json' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["codex_pid"])'
}

# wait until the live bridge advertises the naming capability
for _ in $(seq 1 120); do
  ctrl="$(curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
rows=rows.get("sessions") if isinstance(rows,dict) else rows
print(next((str(r.get("session_name_control")) for r in rows or [] if r.get("session_id")), ""))')"
  [[ "$ctrl" == "True" ]] && break
  sleep 0.25
done
[[ "${ctrl:-}" == "True" ]] || fail "bridge never advertised session_name_control"

# --- Step 1: baseline — the agent is healthy, the rename lands ---------------
rename "healthy-name" 1
python3 - "$artifacts/rename-1.json" <<'PY' || fail "baseline rename did not propagate"
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
prop = body.get("name_propagation") or {}
assert body.get("alias") == "healthy-name", body
assert prop.get("ok") is True and prop.get("reason") == "applied", prop
PY
for _ in $(seq 1 40); do
  [[ "$(log_names)" == *'"healthy-name"'* ]] && break; sleep 0.25
done
[[ "$(log_names)" == *'"healthy-name"'* ]] || fail "baseline name never reached the Pi log"
echo "STEP 1 ok: healthy agent -> rename propagated (applied)"

# --- Step 2: freeze the agent, then rename -----------------------------------
pid="$(pi_pid)"
[[ -n "$pid" ]] || fail "could not read the agent pid from the sidecar"
sudo docker exec "$container" kill -STOP "$pid"
echo "STEP 2: agent pid ${pid} frozen (SIGSTOP); its listening socket still accepts"

t="$(curl -sS -o "$artifacts/rename-stalled.json" -w '%{time_total}' \
  -b "$cookie" -H 'Content-Type: application/json' \
  --data '{"name":"during-stall"}' \
  "http://127.0.0.1:${port}/api/sessions/${session_id}/rename")"
echo "STEP 2: rename HTTP latency while agent frozen: ${t}s"
python3 - "$artifacts/rename-stalled.json" "$t" "$artifacts/step2.json" <<'PY' || fail "expected a reported propagation failure while the agent was frozen"
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
lat = float(sys.argv[2])
prop = body.get("name_propagation") or {}
# The rename is recorded (alias changed) yet the backend title reports a failure.
assert body.get("ok") is True, body
assert body.get("alias") == "during-stall", body
assert prop.get("ok") is not True, prop
assert prop.get("reason") in ("error", "transport"), prop
assert lat >= 1.2, lat
json.dump({"latency_s": lat, "alias": body.get("alias"), "propagation": prop}, open(sys.argv[3], "w"), indent=2)
print("STEP 2: API says the rename succeeded (alias=during-stall) AND reports", prop.get("reason"), prop.get("detail"))
PY

names_after_stall="$(log_names)"
[[ "$names_after_stall" != *'"during-stall"'* ]] || fail "agent recorded a title while frozen (impossible)"
echo "STEP 2 ok: agent still has its old title: ${names_after_stall}"

# The failure is recorded against the session's name projection.
sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/session_names.json' > "$artifacts/session_names-after-stall.json" 2>/dev/null || true
python3 - "$artifacts/session_names-after-stall.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
applied = data.get("applied") or {}
fails = {k: v for k, v in applied.items() if v.get("state") == "failed"}
print("STEP 2: recorded projection failures:", json.dumps(fails))
PY

# --- Step 3: unfreeze — the write was delivered all along --------------------
# The broker gave up on the reply, but the request bytes were already handed to
# the agent's socket. The moment the agent runs again it applies the name, with
# no rename and no further request. The op reported as failed had, in fact,
# been carried out. Nothing here talks to the server; only the agent log is read.
sudo docker exec "$container" kill -CONT "$pid"
applied=0
for _ in $(seq 1 40); do
  if [[ "$(log_names)" == *'"during-stall"'* ]]; then applied=1; break; fi
  sleep 0.25
done
[[ "$applied" == "1" ]] || fail "name was never applied after the agent resumed"
echo "STEP 3 ok: the name the API reported as failed appeared with no user action"

sudo docker exec "$container" sh -c 'cat "$HOME"/.local/share/codoxear/session_names.json' > "$artifacts/session_names-final.json" 2>/dev/null || true
python3 - "$artifacts/session_names-final.json" <<'PY'
import json, sys
rec = next(iter((json.load(open(sys.argv[1], encoding="utf-8")).get("applied") or {}).values()), {})
print("STEP 3: codoxear still records the rename as failed:",
      json.dumps({k: rec.get(k) for k in ("state", "name", "reason", "detail")}))
PY


sudo docker logs "$container" > "$artifacts/server.log" 2>&1 || true
grep 'session-name:' "$artifacts/server.log" > "$artifacts/server-name-lines.txt" 2>/dev/null || true
echo "--- server projection failure log ---"
cat "$artifacts/server-name-lines.txt" 2>/dev/null || true

echo "PASS: reproduced a false rename failure caused by a busy agent process"
echo "artifacts=$artifacts"
