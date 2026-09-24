#!/usr/bin/env bash
# Docker-isolated behavioral verifier for server-synced composer drafts
# (tombstone semantics).
#
# Unlike docker_verify.sh / docker_ui_flows.sh (which build from a committed
# git archive), this verifier builds its image from the CURRENT WORKING
# TREE, because the draft-sync feature under verification is uncommitted.
# The browser bundle is rebuilt with the same esbuild invocation deploy.sh
# uses, but only inside the throwaway build context: the checkout's tracked
# codoxear/static/dist/app.bundle.js is never modified.
#
# Verified behavior (tombstone model: deleting a draft writes
# {text:"", updated_ts:<server wall clock>} — never ts 0; only a
# never-drafted session reports 0):
#   API: GET/POST /api/sessions/<sid>/draft contract, second-client
#        overwrite, sessions-row draft_updated_ts, empty-text delete as a
#        timestamped tombstone, auth/validation/size errors, oversize blank
#        never 413s, never-drafted session shows ts 0 / row 0.0.
#   UI regression case: context A types a draft and leaves the session
#        open; fresh context B sees it and SENDS it through the real UI.
#        The server draft must become a tombstone (text "", ts > 0); B's
#        composer stays empty and never re-pushes; A — still open — clears
#        itself through the pull-if-clean path within a poll cycle or two;
#        nothing ever re-POSTs the sent text (the tombstone ts never
#        advances afterwards), and a reload + re-select in A must not
#        resurrect the sent draft.
#   Basics still passing: cross-context appearance (A types → fresh B
#        sees it), pull-if-clean of an edit (A edits → open clean B
#        receives it).
#   Persistence: with a tombstone present, restarting ONLY the server
#        inside the container keeps the tombstone (same ts, not 0), and
#        browser clients reloading + re-selecting never re-push it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_DRAFT_VERIFY_PORT:-19673}"
password="${CODOXEAR_DRAFT_VERIFY_PASSWORD:-docker-draft-verify-password}"
image="codoxear-draft-verify:worktree"
container="codoxear-draft-verify-${port}"
source_pi_agent="$HOME/.pi/agent"
root="$(mktemp -d /tmp/codoxear-draft-verify.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-draft-verify-results.XXXXXX)"
home_dir="$root/home"
browser_a="codoxear-draft-a-$$"
browser_b="codoxear-draft-b-$$"
readonly root artifacts home_dir browser_a browser_b
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override; unset CODEXEAR_APP_DIR and CODEX_WEB_APP_DIR" >&2
  exit 2
fi
if [[ ! -r "$source_pi_agent/models.json" || ! -r "$source_pi_agent/settings.json" ]]; then
  echo "Pi models.json and settings.json are required under $source_pi_agent" >&2
  exit 2
fi
if ! command -v agent-browser >/dev/null; then
  echo "agent-browser is required" >&2
  exit 2
fi

ab_a() { AGENT_BROWSER_SESSION="$browser_a" agent-browser "$@"; }
ab_b() { AGENT_BROWSER_SESSION="$browser_b" agent-browser "$@"; }

capture_container_diagnostics() {
  "${docker[@]}" logs "$container" > "$artifacts/server-container.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc '
    printf "HOME=%s\n" "$HOME"
    printf -- "--- server.pid ---\n"; cat "$HOME/server.pid" 2>/dev/null || true
    printf -- "--- server.log (tail) ---\n"; tail -n 80 "$HOME/server.log" 2>/dev/null || true
    printf -- "--- socks ---\n"
    find "$HOME/.local/share/codoxear/socks" -maxdepth 1 -type f -name "*.json" -printf "%p\n" 2>/dev/null
    printf -- "--- session_drafts.json ---\n"
    cat "$HOME/.local/share/codoxear/session_drafts.json" 2>/dev/null || true
  ' > "$artifacts/isolation-and-drafts.txt" 2>&1 || true
  "${docker[@]}" exec "$container" sh -lc 'cat "$HOME/pi-bootstrap.log" "$HOME/broker.log"' > "$artifacts/pi-and-broker.log" 2>&1 || true
}

cleanup() {
  capture_container_diagnostics
  ab_a close >/dev/null 2>&1 || true
  ab_b close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  "${docker[@]}" rmi "$image" >/dev/null 2>&1 || true
  rm -rf "$root"
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
    if [[ -r "$source_pi_agent/$name" ]]; then
      install -m 600 "$source_pi_agent/$name" "$home_dir/.pi/agent/$name"
    fi
  done
}

json_password() {
  python3 -c 'import json, sys; print(json.dumps({"password": sys.argv[1]}))' "$password"
}

wait_for_server() {
  for _ in $(seq 1 120); do
    if [[ "$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)" == "401" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Stop only the server process (exact PID recorded at launch) and start a
# fresh one with the same environment; the broker and Pi stay alive.
restart_server_in_container() {
  "${docker[@]}" exec "$container" sh -c '
    set -eu
    pid="$(cat "$HOME/server.pid")"
    kill "$pid"
    i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 150 ]; do sleep 0.2; i=$((i+1)); done
    if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid"; sleep 1; fi
    cd /workspace
    export PYTHONPATH=/workspace
    nohup python3 -m codoxear.server >>"$HOME/server.log" 2>&1 &
    echo $! >"$HOME/server.pid"
  ' || return 1
  wait_for_server
}

api_get() { # jar path out
  curl -sS -b "$1" -o "$3" -w '%{http_code}' "http://127.0.0.1:${port}$2"
}
api_post() { # jar path body out
  curl -sS -b "$1" -H 'Content-Type: application/json' -o "$4" -w '%{http_code}' --data "$3" "http://127.0.0.1:${port}$2"
}
api_post_file() { # jar path bodyfile out
  curl -sS -b "$1" -H 'Content-Type: application/json' -o "$4" -w '%{http_code}' --data @"$3" "http://127.0.0.1:${port}$2"
}

wait_for_pi_session() { # outfile
  local sessions="$1"
  for _ in $(seq 1 120); do
    local status
    status="$(api_get "$root/cookiesA.txt" /api/sessions "$sessions")" || true
    if [[ "$status" == "200" ]] && python3 - "$sessions" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
raise SystemExit(0 if any(isinstance(r, dict) and r.get("agent_backend") == "pi" and isinstance(r.get("session_id"), str) for r in rows) else 1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_idle() {
  for _ in $(seq 1 60); do
    api_get "$root/cookiesA.txt" /api/sessions "$artifacts/api-sessions-live.json" >/dev/null || true
    if python3 - "$artifacts/api-sessions-live.json" "$session_id" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("sessions") if isinstance(payload, dict) else []
row = next((r for r in rows if isinstance(r, dict) and r.get("session_id") == sys.argv[2]), {})
raise SystemExit(0 if not row.get("busy") and not row.get("queue_len") else 1)
PY
    then
      return 0
    fi
    sleep 1
  done
  return 1
}

composer_value() { # browser_fn out  -- writes {value, err}
  local out="$1"
  shift
  "$@" eval '(() => { const node = document.querySelector("#msg"); return { value: node ? node.value : null, active: Boolean(document.querySelector("#sessions .session.active")) }; })()' --json > "$out" 2>&1 || true
}

poll_composer_until() { # expected_text attempts sleep_secs out browser_fn...
  local expected="$1" attempts="$2" sleep_secs="$3" out="$4"
  shift 4
  for _ in $(seq 1 "$attempts"); do
    composer_value "$out" "$@"
    if python3 -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8"))["data"]["result"]; raise SystemExit(0 if d.get("value") == sys.argv[2] else 1)' "$out" "$expected" 2>/dev/null; then
      return 0
    fi
    sleep "$sleep_secs"
  done
  return 1
}

# Sample the composer every second for N seconds; record every sample.
sample_composer_for() { # seconds out browser_fn...
  local seconds="$1" out="$2"
  shift 2
  : > "$out"
  local i
  for i in $(seq 0 "$seconds"); do
    composer_value "$artifacts/.sample-tmp.json" "$@"
    python3 -c 'import json,sys; print(json.dumps({"t": sys.argv[1], "value": json.load(open(sys.argv[2],encoding="utf-8"))["data"]["result"].get("value")}))' "$i" "$artifacts/.sample-tmp.json" >> "$out" 2>/dev/null || echo "{\"t\": $i, \"value\": \"sample-error\"}" >> "$out"
    [[ "$i" -eq "$seconds" ]] && break
    sleep 1
  done
}

localstate_eval='(() => { const sid = document.querySelector("#sessions .session.active")?.dataset.sessionId || null; return { sid, draft: sid ? localStorage.getItem("codexweb.draft." + sid) : null, companion: sid ? localStorage.getItem("codexweb.draft." + sid + ".server_ts") : null }; })()'

copy_pi_config

# ---- Build context from the WORKING TREE (tracked files at their current
# working-tree state + untracked non-ignored files, minus local-only dirs).
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
install -m 644 "$repo_root/docker/verify.Dockerfile" "$build_context/docker/verify.Dockerfile"
# Rebuild the browser bundle from the working-tree ESM source INTO THE CONTEXT
# ONLY (same invocation as scripts/deploy.sh). The checkout stays untouched.
npx --yes esbuild "$build_context/codoxear/static/app.js" --bundle --minify \
  --outfile="$build_context/codoxear/static/dist/app.bundle.js" --format=esm \
  > "$artifacts/esbuild.log" 2>&1 || fail "esbuild bundle rebuild failed; see $artifacts/esbuild.log"

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
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" bash -c '
    set -eu
    cd /workspace
    export PYTHONPATH=/workspace
    pi --offline --no-extensions --no-skills --no-prompt-templates --no-context-files \
      --session-id docker-draft-verify-session -p "Initialize isolated draft-verify session." \
      >"$HOME/pi-bootstrap.log" 2>&1 || true
    session_log="$(find "$HOME/.pi/agent/sessions" -type f -name "*docker-draft-verify-session.jsonl" -print -quit)"
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
    test -n "${meta:-}" && grep -Eq "\"session_id\"[[:space:]]*:[[:space:]]*\"[^\"]+\"" "$meta"
    # Server runs as a restartable background process (exact PID recorded);
    # the container itself just stays alive.
    python3 -m codoxear.server >"$HOME/server.log" 2>&1 &
    echo $! >"$HOME/server.pid"
    exec sleep infinity
  ' > "$artifacts/container-id.txt" || fail "container start failed"

wait_for_server || fail "isolated server did not become reachable"
curl -sS -c "$root/cookiesA.txt" -H 'Content-Type: application/json' --data "$(json_password)" \
  "http://127.0.0.1:${port}/api/login" > "$artifacts/loginA.json" || fail "API login (client A) failed"
curl -sS -c "$root/cookiesB.txt" -H 'Content-Type: application/json' --data "$(json_password)" \
  "http://127.0.0.1:${port}/api/login" > "$artifacts/loginB.json" || fail "API login (client B) failed"
wait_for_pi_session "$artifacts/api-sessions-initial.json" || fail "offline Pi session did not appear"

session_id="$(python3 - "$artifacts/api-sessions-initial.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("sessions", [])
for row in rows:
    if isinstance(row, dict) and row.get("agent_backend") == "pi" and isinstance(row.get("session_id"), str):
        print(row["session_id"]); break
else:
    raise SystemExit("no Pi session id")
PY
)" || fail "could not select Pi session"
printf '%s\n' "$session_id" > "$artifacts/session-id.txt"
draft_base="/api/sessions/${session_id}/draft"
card_selector="#sessions .session[data-session-id=\"${session_id}\"]"

# ============================ API PHASE ============================
alpha_api="api-draft-alpha-$(date +%s)"
beta_api="api-draft-beta-$(date +%s)"

printf '%s\n' "$(api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/api-01-initial.json")" > "$artifacts/api-01-initial.status"
api_get "$root/cookiesA.txt" /api/sessions "$artifacts/api-01b-sessions-neverdrafted.json" > "$artifacts/api-01b-sessions-neverdrafted.status"
printf '%s\n' "$(api_post "$root/cookiesA.txt" "$draft_base" "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$alpha_api")" "$artifacts/api-02-post-a.json")" > "$artifacts/api-02-post-a.status"
printf '%s\n' "$(api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/api-03-get-a.json")" > "$artifacts/api-03-get-a.status"
printf '%s\n' "$(api_post "$root/cookiesB.txt" "$draft_base" "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$beta_api")" "$artifacts/api-04-post-b.json")" > "$artifacts/api-04-post-b.status"
printf '%s\n' "$(api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/api-05-get-a-after-b.json")" > "$artifacts/api-05-get-a-after-b.status"
api_get "$root/cookiesA.txt" /api/sessions "$artifacts/api-06-sessions.json" > "$artifacts/api-06-sessions.status"
# Deletion is a tombstone: ts of the delete response is a NEW server ts.
printf '%s\n' "$(api_post "$root/cookiesB.txt" "$draft_base" '{"text":""}' "$artifacts/api-07-delete.json")" > "$artifacts/api-07-delete.status"
printf '%s\n' "$(api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/api-08-get-after-delete.json")" > "$artifacts/api-08-get-after-delete.status"
api_get "$root/cookiesA.txt" /api/sessions "$artifacts/api-09-sessions.json" > "$artifacts/api-09-sessions.status"
# unauthenticated
curl -sS -o "$artifacts/api-10-unauth-get.json" -w '%{http_code}' "http://127.0.0.1:${port}${draft_base}" > "$artifacts/api-10-unauth-get.status"
curl -sS -o "$artifacts/api-11-unauth-post.json" -w '%{http_code}' -H 'Content-Type: application/json' --data '{"text":"x"}' "http://127.0.0.1:${port}${draft_base}" > "$artifacts/api-11-unauth-post.status"
# validation
printf '%s\n' "$(api_post "$root/cookiesA.txt" "$draft_base" '{"text":5}' "$artifacts/api-12-badtype.json")" > "$artifacts/api-12-badtype.status"
printf '%s\n' "$(api_get "$root/cookiesA.txt" "/api/sessions/no-such-session/draft" "$artifacts/api-13-unknown-get.json")" > "$artifacts/api-13-unknown-get.status"
printf '%s\n' "$(api_post "$root/cookiesA.txt" "/api/sessions/no-such-session/draft" '{"text":"x"}' "$artifacts/api-14-unknown-post.json")" > "$artifacts/api-14-unknown-post.status"
# size limits: 256 KiB is allowed, 256 KiB + 1 is rejected
python3 - <<'PY' > "$root/oversize-plus-one.json"
import json
print(json.dumps({"text": "a" * (256 * 1024 + 1)}))
PY
python3 - <<'PY' > "$root/oversize-exact.json"
import json
print(json.dumps({"text": "a" * (256 * 1024)}))
PY
printf '%s\n' "$(api_post_file "$root/cookiesA.txt" "$draft_base" "$root/oversize-plus-one.json" "$artifacts/api-15-oversize.json")" > "$artifacts/api-15-oversize.status"
printf '%s\n' "$(api_post_file "$root/cookiesA.txt" "$draft_base" "$root/oversize-exact.json" "$artifacts/api-16-boundary.json")" > "$artifacts/api-16-boundary.status"
# whitespace-only deletes; an oversize BLANK (300 KiB of spaces) must never
# 413 — blank clears bypass the byte cap by normalizing to a tombstone.
python3 - <<'PY' > "$root/oversize-blank.json"
import json
print(json.dumps({"text": " " * (300 * 1024)}))
PY
printf '%s\n' "$(api_post "$root/cookiesA.txt" "$draft_base" '{"text":"   "}' "$artifacts/api-17-whitespace.json")" > "$artifacts/api-17-whitespace.status"
printf '%s\n' "$(api_post_file "$root/cookiesA.txt" "$draft_base" "$root/oversize-blank.json" "$artifacts/api-17b-oversize-blank.json")" > "$artifacts/api-17b-oversize-blank.status"
printf '%s\n' "$(api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/api-18-final.json")" > "$artifacts/api-18-final.status"
api_get "$root/cookiesA.txt" /api/sessions "$artifacts/api-19-sessions-final.json" > "$artifacts/api-19-sessions-final.status"

python3 - "$artifacts" "$session_id" "$alpha_api" "$beta_api" <<'PY' > "$artifacts/api-checks.json"
import json, sys
from pathlib import Path

artifacts = Path(sys.argv[1])
session_id, alpha, beta = sys.argv[2:5]

def read_json(name):
    try:
        return json.loads((artifacts / name).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc)}

def read_status(name):
    try:
        return (artifacts / name).read_text(encoding="utf-8").strip()
    except Exception as exc:
        return f"read_error: {exc}"

def row_of(name):
    payload = read_json(name)
    rows = payload.get("sessions") if isinstance(payload, dict) else []
    return next((r for r in rows if isinstance(r, dict) and r.get("session_id") == session_id), {})

initial = read_json("api-01-initial.json")
post_a = read_json("api-02-post-a.json")
get_a = read_json("api-03-get-a.json")
post_b = read_json("api-04-post-b.json")
get_a_after_b = read_json("api-05-get-a-after-b.json")
delete = read_json("api-07-delete.json")
after_delete = read_json("api-08-get-after-delete.json")
final = read_json("api-18-final.json")

def row_ts(name):
    value = row_of(name).get("draft_updated_ts")
    return None if value is None else float(value)

ts_a = float(post_a.get("updated_ts") or 0)
ts_b = float(post_b.get("updated_ts") or 0)
ts_tomb = float(delete.get("updated_ts") or 0)
ts_ws = float(read_json("api-17-whitespace.json").get("updated_ts") or 0)
ts_blank = float(read_json("api-17b-oversize-blank.json").get("updated_ts") or 0)
row_ts_final = row_ts("api-19-sessions-final.json")

checks = {
    "initial_neverdrafted_get_zero": read_status("api-01-initial.status") == "200" and initial.get("ok") is True and initial.get("text") == "" and float(initial.get("updated_ts") or 0) == 0,
    "neverdrafted_row_zero": row_ts("api-01b-sessions-neverdrafted.json") == 0.0,
    "post_a_returns_ts": read_status("api-02-post-a.status") == "200" and post_a.get("ok") is True and ts_a > 0,
    "get_echoes_a": read_status("api-03-get-a.status") == "200" and get_a.get("text") == alpha and abs(float(get_a.get("updated_ts") or 0) - ts_a) < 1e-6,
    "post_b_overwrites": read_status("api-04-post-b.status") == "200" and ts_b >= ts_a,
    "a_sees_b_text": read_status("api-05-get-a-after-b.status") == "200" and get_a_after_b.get("text") == beta,
    "row_ts_tracks_latest": row_ts("api-06-sessions.json") is not None and abs(row_ts("api-06-sessions.json") - ts_b) < 1e-6,
    "delete_returns_tombstone_ts": read_status("api-07-delete.status") == "200" and delete.get("ok") is True and ts_tomb > 0 and ts_tomb >= ts_b,
    "get_after_delete_keeps_tombstone": read_status("api-08-get-after-delete.status") == "200" and after_delete.get("text") == "" and abs(float(after_delete.get("updated_ts") or 0) - ts_tomb) < 1e-6,
    "row_carries_tombstone_ts": row_ts("api-09-sessions.json") is not None and abs(row_ts("api-09-sessions.json") - ts_tomb) < 1e-6,
    "unauth_get_401": read_status("api-10-unauth-get.status") == "401",
    "unauth_post_401": read_status("api-11-unauth-post.status") == "401",
    "non_string_text_400": read_status("api-12-badtype.status") == "400",
    "unknown_session_get_404": read_status("api-13-unknown-get.status") == "404",
    "unknown_session_post_404": read_status("api-14-unknown-post.status") == "404",
    "oversize_413": read_status("api-15-oversize.status") == "413",
    "exact_256k_allowed": read_status("api-16-boundary.status") == "200",
    "whitespace_deletes_as_tombstone": read_status("api-17-whitespace.status") == "200" and ts_ws > 0 and ts_ws >= ts_tomb,
    "oversize_blank_never_413s": read_status("api-17b-oversize-blank.status") == "200" and ts_blank > 0 and ts_blank >= ts_ws,
    "final_state_is_latest_tombstone": read_status("api-18-final.status") == "200" and final.get("text") == "" and abs(float(final.get("updated_ts") or 0) - ts_blank) < 1e-6,
    "final_row_carries_tombstone": row_ts_final is not None and abs(float(row_ts_final) - ts_blank) < 1e-6,
}
observed = {
    "initial": initial,
    "post_a": post_a,
    "get_a": get_a,
    "post_b": post_b,
    "get_a_after_b": get_a_after_b,
    "row_ts_after_b": row_of("api-06-sessions.json").get("draft_updated_ts"),
    "delete": delete,
    "after_delete": after_delete,
    "row_ts_after_delete": row_of("api-09-sessions.json").get("draft_updated_ts"),
    "whitespace_delete": read_json("api-17-whitespace.json"),
    "oversize_blank": read_json("api-17b-oversize-blank.json"),
    "final": final,
    "final_row_ts": row_ts_final,
    "oversize_body": read_json("api-15-oversize.json"),
    "badtype_body": read_json("api-12-badtype.json"),
}
out = {"checks": checks, "observed": observed, "pass": all(checks.values())}
print(json.dumps(out, indent=2))
PY

python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["pass"] else 1)' "$artifacts/api-checks.json" || fail "API phase failed; see $artifacts/api-checks.json"

# ============================ UI PHASE ============================
app_url="http://127.0.0.1:${port}/"

# Context A: open, login, select the session, and STAY OPEN on it.
ab_a open "$app_url" > "$artifacts/ui-a-open.txt" 2>&1 || fail "context A could not open app"
ab_a fill '#pw' "$password" > "$artifacts/ui-a-login-fill.txt" 2>&1 || fail "context A login form did not render"
ab_a click '#loginBtn' > "$artifacts/ui-a-login-click.txt" 2>&1 || fail "context A login click failed"
ab_a wait 2500 > /dev/null 2>&1 || true
ab_a click "$card_selector" > "$artifacts/ui-a-select.txt" 2>&1 || fail "context A could not click the session card"
ab_a wait 1500 > /dev/null 2>&1 || true
composer_value "$artifacts/ui-a-01-initial.json" ab_a
a_selection_eval='(() => ({ active: Boolean(document.querySelector("#sessions .session.active")), activeIsTarget: document.querySelector("#sessions .session.active")?.dataset.sessionId === '"'"$session_id"'"', bootstrapped: window.__codoxearAppBootstrapped === true }))()'
ab_a eval "$a_selection_eval" --json > "$artifacts/ui-a-01-selection.json" 2>&1 || true

alpha_ui="ui-draft-alpha-from-context-A-$(date +%s)"
beta_ui="ui-draft-beta-edit-from-context-A-$(date +%s)"

# A types a draft (does not send); the debounced upload lands server-side.
ab_a fill '#msg' "$alpha_ui" > "$artifacts/ui-a-type-alpha.txt" 2>&1 || fail "context A composer fill failed"
sleep 2.5
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-a-02-server-after-alpha.json" > "$artifacts/ui-a-02-server-after-alpha.status"
ab_a eval "$localstate_eval" --json > "$artifacts/ui-a-02-localstate.json" 2>&1 || true

# Context B: fresh browser profile => fresh localStorage. Same session.
ab_b open "$app_url" > "$artifacts/ui-b-open.txt" 2>&1 || fail "context B could not open app"
ab_b fill '#pw' "$password" > "$artifacts/ui-b-login-fill.txt" 2>&1 || fail "context B login form did not render"
ab_b click '#loginBtn' > "$artifacts/ui-b-login-click.txt" 2>&1 || fail "context B login click failed"
ab_b wait 2500 > /dev/null 2>&1 || true
ab_b click "$card_selector" > "$artifacts/ui-b-select.txt" 2>&1 || fail "context B could not click the session card"
ab_b wait 1500 > /dev/null 2>&1 || true
cross_appeared="no"
if poll_composer_until "$alpha_ui" 20 1 "$artifacts/ui-b-03-composer.json" ab_b; then
  cross_appeared="yes"
fi
ab_b screenshot --full "$artifacts/ui-b-03-cross-context-draft.png" > "$artifacts/ui-b-03-screenshot.txt" 2>&1 || true
ab_b eval "$localstate_eval" --json > "$artifacts/ui-b-03-localstate.json" 2>&1 || true

# Live pull-if-clean: A edits the draft; B (open, unedited) receives beta.
ab_a fill '#msg' "$beta_ui" > "$artifacts/ui-a-type-beta.txt" 2>&1 || fail "context A beta fill failed"
sleep 2.5
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-a-04-server-after-beta.json" > "$artifacts/ui-a-04-server-after-beta.status"
live_pull="no"
if poll_composer_until "$beta_ui" 30 2 "$artifacts/ui-b-04-live-pull.json" ab_b; then
  live_pull="yes"
fi

# ------------------ REGRESSION CASE: B sends A's draft ------------------
wait_for_idle || true
send_started=$(date +%s)
ab_b click '#sendBtn' > "$artifacts/ui-b-send-click.txt" 2>&1 || fail "context B send button unavailable"
ab_b wait 1200 > /dev/null 2>&1 || true
# If the busy send-choice dialog intercepted, confirm "send now".
ab_b eval '(() => { const c = document.querySelector("#sendChoice"); return Boolean(c && getComputedStyle(c).display === "flex"); })()' --json > "$artifacts/ui-b-send-choice-state.json" 2>&1 || true
if python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["data"]["result"] is True else 1)' "$artifacts/ui-b-send-choice-state.json" 2>/dev/null; then
  ab_b click '#sendChoiceNowBtn' > "$artifacts/ui-b-send-choice-now.txt" 2>&1 || true
  ab_b wait 1200 > /dev/null 2>&1 || true
fi
tombstone_landed=$(date +%s)
composer_value "$artifacts/ui-b-05-after-send.json" ab_b
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-b-05-server-after-send.json" > "$artifacts/ui-b-05-server-after-send.status"
api_get "$root/cookiesA.txt" /api/sessions "$artifacts/ui-b-05-sessions-after-send.json" > /dev/null
ab_b eval "$localstate_eval" --json > "$artifacts/ui-b-05-localstate.json" 2>&1 || true
ab_b screenshot --full "$artifacts/ui-b-05-after-send.png" > "$artifacts/ui-b-05-screenshot.txt" 2>&1 || true

# B's composer must stay empty and B must never re-push: sample the
# composer for 10s, then confirm the server tombstone ts did not advance.
sample_composer_for 10 "$artifacts/ui-b-06-stay-empty-samples.jsonl" ab_b
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-b-06-server-after-window.json" > "$artifacts/ui-b-06-server-after-window.status"

# A — still open on the session — must clear itself via pull-if-clean
# (sessions poll is 5s visible; allow a couple of poll cycles plus slack).
a_cleared="no"
for attempt in $(seq 1 25); do
  composer_value "$artifacts/ui-a-07-clear-check.json" ab_a
  if python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["data"]["result"].get("value") == "" else 1)' "$artifacts/ui-a-07-clear-check.json" 2>/dev/null; then
    a_cleared="yes"
    a_clear_elapsed=$(( $(date +%s) - tombstone_landed ))
    break
  fi
  sleep 1
done
ab_a screenshot --full "$artifacts/ui-a-07-after-clear.png" > "$artifacts/ui-a-07-screenshot.txt" 2>&1 || true
ab_a eval "$localstate_eval" --json > "$artifacts/ui-a-07-localstate.json" 2>&1 || true

# Settle: whatever clients do next, the server must still hold the exact
# same tombstone (ts never advanced => nobody ever re-POSTed old text).
sleep 8
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-a-08-server-settled.json" > "$artifacts/ui-a-08-server-settled.status"

# Reload + re-select in A: the sent draft must not resurrect anywhere.
ab_a open "$app_url" > "$artifacts/ui-a-reload.txt" 2>&1 || fail "context A reload failed"
ab_a wait 3500 > /dev/null 2>&1 || true
ab_a click "$card_selector" > "$artifacts/ui-a-reselect.txt" 2>&1 || fail "context A could not re-select the session card"
ab_a wait 3000 > /dev/null 2>&1 || true
composer_value "$artifacts/ui-a-09-after-reload-reselect.json" ab_a
ab_a eval "$localstate_eval" --json > "$artifacts/ui-a-09-localstate.json" 2>&1 || true
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-a-09-server-after-reload.json" > "$artifacts/ui-a-09-server-after-reload.status"
ab_a screenshot --full "$artifacts/ui-a-09-after-reload-reselect.png" > "$artifacts/ui-a-09-screenshot.txt" 2>&1 || true

# ==================== PERSISTENCE: server-only restart ====================
# The draft is a tombstone at this point. Restart ONLY the server process
# (exact recorded PID); the broker and Pi keep running.
restart_started=$(date +%s)
restart_server_in_container || fail "in-container server restart failed"
wait_for_pi_session "$artifacts/api-20-sessions-after-restart.json" || fail "session list did not recover after server restart"
printf '%s\n' "$(api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/api-21-after-restart.json")" > "$artifacts/api-21-after-restart.status"
api_get "$root/cookiesA.txt" /api/sessions "$artifacts/api-22-sessions-after-restart.json" > "$artifacts/api-22-sessions-after-restart.status"

# A reload + re-select after the restart must not re-push the tombstone.
ab_a open "$app_url" > "$artifacts/ui-a-reload2.txt" 2>&1 || fail "context A post-restart reload failed"
ab_a wait 3500 > /dev/null 2>&1 || true
ab_a click "$card_selector" > "$artifacts/ui-a-reselect2.txt" 2>&1 || fail "context A could not re-select after restart"
ab_a wait 3000 > /dev/null 2>&1 || true
composer_value "$artifacts/ui-a-10-post-restart.json" ab_a
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-a-10-server-post-restart.json" > "$artifacts/ui-a-10-server-post-restart.status"

# Same for B (the client that sent): reload + re-select, no re-push.
ab_b open "$app_url" > "$artifacts/ui-b-reload.txt" 2>&1 || fail "context B post-restart reload failed"
ab_b wait 3500 > /dev/null 2>&1 || true
ab_b click "$card_selector" > "$artifacts/ui-b-reselect.txt" 2>&1 || fail "context B could not re-select after restart"
ab_b wait 3000 > /dev/null 2>&1 || true
composer_value "$artifacts/ui-b-10-post-restart.json" ab_b
api_get "$root/cookiesA.txt" "$draft_base" "$artifacts/ui-b-10-server-post-restart.json" > "$artifacts/ui-b-10-server-post-restart.status"

ab_a errors --json > "$artifacts/ui-a-errors.json" 2>&1 || true
ab_b errors --json > "$artifacts/ui-b-errors.json" 2>&1 || true

python3 - "$artifacts" "$session_id" "$alpha_ui" "$beta_ui" "$cross_appeared" "$live_pull" "$a_cleared" "${a_clear_elapsed:--1}" <<'PY' > "$artifacts/ui-checks.json"
import json, sys
from pathlib import Path

artifacts = Path(sys.argv[1])
session_id, alpha, beta, cross_appeared, live_pull, a_cleared = sys.argv[2:8]
a_clear_elapsed = int(sys.argv[8])

def read_json(name):
    try:
        return json.loads((artifacts / name).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc)}

def read_lines(name):
    try:
        return [json.loads(line) for line in (artifacts / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []

def browser_result(name):
    payload = read_json(name)
    if isinstance(payload, dict) and payload.get("success") is True:
        data = payload.get("data")
        if isinstance(data, dict) and "result" in data:
            return data["result"]
    return payload

def status(name):
    try:
        return (artifacts / name).read_text(encoding="utf-8").strip()
    except Exception as exc:
        return f"read_error: {exc}"

def row_of(name):
    payload = read_json(name)
    rows = payload.get("sessions") if isinstance(payload, dict) else []
    return next((r for r in rows if isinstance(r, dict) and r.get("session_id") == session_id), {})

def localstate(name):
    result = browser_result(name)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return {}
    return result if isinstance(result, dict) else {}

def ts_of(name):
    return float(read_json(name).get("updated_ts") or 0)

server_after_alpha = read_json("ui-a-02-server-after-alpha.json")
server_after_beta = read_json("ui-a-04-server-after-beta.json")
after_send = read_json("ui-b-05-server-after-send.json")
after_window = read_json("ui-b-06-server-after-window.json")
settled = read_json("ui-a-08-server-settled.json")
after_a_reload = read_json("ui-a-09-server-after-reload.json")
tomb_ts = float(after_send.get("updated_ts") or 0)
beta_ts = float(server_after_beta.get("updated_ts") or 0)
b_stay_empty = read_lines("ui-b-06-stay-empty-samples.jsonl")
b_all_empty = bool(b_stay_empty) and all(s.get("value") == "" for s in b_stay_empty)

checks = {
    "a_selected_session": browser_result("ui-a-01-selection.json").get("activeIsTarget") is True,
    "a_initial_composer_empty": browser_result("ui-a-01-initial.json").get("value") == "",
    "debounced_upload_landed": status("ui-a-02-server-after-alpha.status") == "200" and server_after_alpha.get("text") == alpha,
    "cross_context_appearance": cross_appeared == "yes" and browser_result("ui-b-03-composer.json").get("value") == alpha,
    "beta_uploaded": status("ui-a-04-server-after-beta.status") == "200" and server_after_beta.get("text") == beta,
    "live_pull_if_clean": live_pull == "yes" and browser_result("ui-b-04-live-pull.json").get("value") == beta,
    # ---- regression case ----
    "regression_send_wrote_tombstone": status("ui-b-05-server-after-send.status") == "200" and after_send.get("text") == "" and tomb_ts > 0 and tomb_ts > beta_ts,
    "regression_row_carries_tombstone": abs(float(row_of("ui-b-05-sessions-after-send.json").get("draft_updated_ts") or -1) - tomb_ts) < 1e-6,
    "regression_b_composer_cleared_by_send": browser_result("ui-b-05-after-send.json").get("value") == "",
    "regression_b_composer_stays_empty": b_all_empty,
    "regression_b_never_repushes": status("ui-b-06-server-after-window.status") == "200" and after_window.get("text") == "" and abs(float(after_window.get("updated_ts") or 0) - tomb_ts) < 1e-6,
    "regression_a_clears_while_open": a_cleared == "yes" and browser_result("ui-a-07-clear-check.json").get("value") == "",
    "regression_a_never_reposts_old_text": status("ui-a-08-server-settled.status") == "200" and settled.get("text") == "" and abs(float(settled.get("updated_ts") or 0) - tomb_ts) < 1e-6,
    "regression_no_resurrection_after_reload_reselect": status("ui-a-09-server-after-reload.status") == "200" and browser_result("ui-a-09-after-reload-reselect.json").get("value") == "" and after_a_reload.get("text") == "" and abs(float(after_a_reload.get("updated_ts") or 0) - tomb_ts) < 1e-6,
    "regression_a_local_draft_gone": localstate("ui-a-09-localstate.json").get("draft") in (None, ""),
    # ---- persistence ----
    "persistence_tombstone_survives_restart": status("api-21-after-restart.status") == "200" and ts_of("api-21-after-restart.json") > 0 and read_json("api-21-after-restart.json").get("text") == "" and abs(ts_of("api-21-after-restart.json") - tomb_ts) < 1e-6,
    "persistence_row_ts_survives_restart": abs(float(row_of("api-22-sessions-after-restart.json").get("draft_updated_ts") or -1) - tomb_ts) < 1e-6,
    "persistence_a_reselect_no_repush": status("ui-a-10-server-post-restart.status") == "200" and browser_result("ui-a-10-post-restart.json").get("value") == "" and abs(float(read_json("ui-a-10-server-post-restart.json").get("updated_ts") or 0) - tomb_ts) < 1e-6,
    "persistence_b_reselect_no_repush": status("ui-b-10-server-post-restart.status") == "200" and browser_result("ui-b-10-post-restart.json").get("value") == "" and abs(float(read_json("ui-b-10-server-post-restart.json").get("updated_ts") or 0) - tomb_ts) < 1e-6,
}
observations = {
    "tombstone_ts": tomb_ts,
    "beta_ts": beta_ts,
    "a_clear_elapsed_seconds": a_clear_elapsed,
    "b_stay_empty_samples": b_stay_empty,
    "a_localstate_after_alpha": localstate("ui-a-02-localstate.json"),
    "b_localstate_at_appearance": localstate("ui-b-03-localstate.json"),
    "b_localstate_after_send": localstate("ui-b-05-localstate.json"),
    "a_localstate_after_clear": localstate("ui-a-07-localstate.json"),
    "a_localstate_after_reload": localstate("ui-a-09-localstate.json"),
    "server_draft_after_b_window": after_window,
    "server_draft_settled": settled,
    "server_draft_after_a_reload": after_a_reload,
    "server_draft_after_restart": read_json("api-21-after-restart.json"),
    "server_draft_after_a_reselect_post_restart": read_json("ui-a-10-server-post-restart.json"),
    "server_draft_after_b_reselect_post_restart": read_json("ui-b-10-server-post-restart.json"),
    "a_page_errors": read_json("ui-a-errors.json"),
    "b_page_errors": read_json("ui-b-errors.json"),
    "expected_alpha": alpha,
    "expected_beta": beta,
}
out = {"checks": checks, "observations": observations, "pass": all(checks.values())}
print(json.dumps(out, indent=2))
PY

python3 - "$artifacts" <<'PY' > "$artifacts/report.json"
import json, sys
from pathlib import Path
artifacts = Path(sys.argv[1])
api = json.loads((artifacts / "api-checks.json").read_text(encoding="utf-8"))
ui = json.loads((artifacts / "ui-checks.json").read_text(encoding="utf-8"))
report = {
    "api": api,
    "ui": ui,
    "pass": api["pass"] and ui["pass"],
}
print(json.dumps(report, indent=2))
PY

cat "$artifacts/api-checks.json"
cat "$artifacts/ui-checks.json"

if ! python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$artifacts/report.json"; then
  echo "FAIL: draft verification failed; artifacts=$artifacts" >&2
  exit 1
fi
printf 'PASS: draft verification artifacts=%s screenshots=%s\n' "$artifacts" "$artifacts/ui-b-03-cross-context-draft.png"
