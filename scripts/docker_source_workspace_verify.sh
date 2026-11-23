#!/usr/bin/env bash
# Docker/browser acceptance for the Codoxear source-workspace feature.
#
# Commit-scoped: builds a Git archive of the requested commit (default HEAD),
# re-bundles the frontend from that archive, and runs the real server + real
# Pi + a real browser against it. Nothing touches the host runtime, the live
# deployment, or port 8743.
#
# Fixture contract (offline, deterministic):
#   - /fixture/canonical.git is a bare clone of exactly the verified commit,
#     mounted read-only. $HOME/.gitconfig redirects the canonical upstream
#     https://github.com/yiwenlu66/codoxear(.git) to it, so the setup clone
#     resolves offline. The container's HTTP(S)_PROXY points at a dead port,
#     so any clone that does not go through the canonical URL fails fast —
#     a HEAD equal to the fixture therefore proves canonical provenance.
#   - Pi runs against a container-local deterministic streaming provider
#     ("LOCAL RESPONSE: <last user message>"), so launched sessions produce
#     real assistant turns without credentials or external traffic.
#
# Stages:
#   1  unauthenticated GET/POST /api/source-workspace -> 401
#   2  authenticated GET before setup: absent shape, no clone on GET
#   3  UI absent state: secondary shortcut visible (desktop), no clone,
#      no session POST; setup affordance offered
#   4  setup failure is explicit and visible in the dialog
#   5  explicit setup succeeds: editable checkout, canonical provenance,
#      contribution skills present, not the serving snapshot
#   6  POST setup is idempotent (no destructive re-clone)
#   7  with the source GET pending, a browse-only open of the cwd list and its
#      dismissal must not defeat the fill: the source path still lands; opening
#      the dialog alone fires no automatic GET
#   8  a cwd typed while the fetch is pending wins over the async landing
#   9  shortcut fills cwd, preserves backend/model, launches nothing; the
#      extended hit lane must not intercept the input, the label, or the header
#  10  manual cwd edit removes the contextual source help
#  11  Help entry closes Help and opens the same dialog prefilled
#  12  a real session launched from the source cwd runs there (sidecar cwd,
#      /proc/<pid>/cwd, deterministic assistant reply)
#  13  an ordinary new session is unaffected
#  14  mobile viewport: shortcut visible, no horizontal overflow
#
# Pi discovery of the contribution skills is validated by the docs-refinement
# harness separately; this script only asserts the checkout carries the
# canonical skill files.
#
# Usage: scripts/docker_source_workspace_verify.sh [commit]
# Results are kept in a /tmp/codoxear-srcws-verify-results.* directory.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="$(git -C "$repo_root" rev-parse --verify "${1:-HEAD}^{commit}")"
short="${commit:0:12}"
port="${CODOXEAR_SOURCE_WS_PORT:-19697}"
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && ((port <= 65535)) && [[ "$port" != 8743 ]] || { echo 'invalid or live port' >&2; exit 2; }
[[ -z "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]] || { echo 'refusing host app-dir override; unset CODEXEAR_APP_DIR and CODEX_WEB_APP_DIR' >&2; exit 2; }
command -v agent-browser >/dev/null || { echo 'agent-browser is required' >&2; exit 2; }
command -v npx >/dev/null || { echo 'npx (esbuild bundling) is required' >&2; exit 2; }

root="$(mktemp -d /tmp/codoxear-srcws-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-srcws-verify-results.XXXXXX)"
home_dir="$root/home"
container="codoxear-srcws-${short}-$$"
browser_session="codoxear-srcws-${short}-$$"
image="${CODOXEAR_SOURCE_WS_IMAGE:-codoxear-srcws-verify:${short}}"
password="docker-srcws-password"
provider_port=19722
readonly repo_root commit short port root artifacts home_dir container browser_session image password provider_port
readonly -a docker=(sudo docker)
browser() { AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"; }
cleanup() {
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  /bin/rm -rf "$root"
}
trap cleanup EXIT
capture_failure_state() {
  # Decisive state must be pulled out of the container/browser before the trap
  # deletes the throwaway HOME. Everything here is best-effort.
  browser screenshot "$artifacts/failure-state.png" > /dev/null 2>&1 || true
  browser eval '(() => { const chat = document.getElementById("chatInner");
    return {chatTail: chat ? chat.innerText.slice(-4000) : null,
      composer: document.getElementById("msg") ? {value: document.getElementById("msg").value, disabled: document.getElementById("msg").disabled} : null,
      sendDisabled: document.getElementById("sendBtn") ? document.getElementById("sendBtn").disabled : null,
      selected: localStorage.getItem("codexweb.selected"),
      bodySample: String(document.body ? document.body.innerText : "").slice(-1500)}; })()' --json > "$artifacts/failure-chat.json" 2>&1 || true
  curl -sS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-at-failure.json" 2>/dev/null || true
  local sid
  sid="$(python3 -c 'import json; rows=json.load(open("'"$artifacts"'/sessions-at-failure.json")).get("sessions") or []; print(rows[0].get("session_id","") if rows else "")' 2>/dev/null || true)"
  if [[ -n "$sid" ]]; then
    curl -sS -b "$cookie" "http://127.0.0.1:${port}/api/sessions/${sid}/messages/tail?limit=100" > "$artifacts/session-tail-at-failure.json" 2>/dev/null || true
  fi
  "${docker[@]}" exec -i "$container" python3 - > "$artifacts/brokers-at-failure.json" 2>&1 <<'PY' || true
import json, os, pathlib
rows = []
for meta_path in sorted(pathlib.Path.home().joinpath(".local/share/codoxear/socks").glob("*.json")):
    meta = json.loads(meta_path.read_text())
    sock = meta_path.with_suffix(".sock")
    state, tail = {}, ""
    try:
        from codoxear.control_socket import call_control_socket
        state = call_control_socket(sock, {"cmd": "state"})
        tail = str(call_control_socket(sock, {"cmd": "tail"}).get("tail", ""))[-3000:]
    except Exception as exc:
        tail = "control socket failed: %s" % exc
    pid = int(meta.get("codex_pid") or 0)
    proc_cwd = os.readlink("/proc/%d/cwd" % pid) if pid > 0 and os.path.exists("/proc/%d/cwd" % pid) else None
    rows.append({"session_id": meta_path.stem, "cwd": meta.get("cwd"), "agent_backend": meta.get("agent_backend"),
                 "proc_cwd": proc_cwd, "state": state, "tail": tail, "log_path": meta.get("log_path")})
session_dir = pathlib.Path.home().joinpath(".pi/agent/sessions")
sessions = sorted(str(p) for p in session_dir.glob("*.jsonl")) if session_dir.is_dir() else []
print(json.dumps({"brokers": rows, "piSessionFiles": sessions}, indent=2))
PY
  "${docker[@]}" exec "$container" sh -c 'cat "$HOME/.verify-provider.log" 2>/dev/null; true' > "$artifacts/provider-requests.log" 2>&1 || true
}
fail() {
  echo "FAIL: $* (artifacts: $artifacts)" >&2
  capture_failure_state
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  "${docker[@]}" exec "$container" sh -c 'cat "$HOME/provider.log" 2>/dev/null; true' > "$artifacts/provider.log" 2>&1 || true
  browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
  browser console --json > "$artifacts/browser-console.json" 2>&1 || true
  exit 1
}

# ---------------------------------------------------------------------------
# Fixtures: one bare canonical clone at the verified commit.
# ---------------------------------------------------------------------------
fixture_sha="$(git -C "$repo_root" rev-parse --verify "${commit}^{commit}")"
mkdir -p "$root/fixture" "$home_dir/.pi/agent"
git clone --quiet "$repo_root" "$root/seed"
git -C "$root/seed" branch --force codoxear-fixture "$commit"
git clone --quiet --bare "$root/seed" "$root/fixture/canonical.git"
git -C "$root/fixture/canonical.git" symbolic-ref HEAD refs/heads/codoxear-fixture
[[ "$(git -C "$root/fixture/canonical.git" rev-parse HEAD)" == "$fixture_sha" ]] || fail 'canonical fixture does not point at the verified commit'
/bin/rm -rf "$root/seed"

# Pi config: deterministic local provider only; no host credentials or
# package auto-install. defaultProjectTrust covers the cloned checkout's
# .agents/skills trust prompt the same way --approve does for web launches.
cat > "$home_dir/.pi/agent/models.json" <<EOF
{"providers": {"local": {
  "baseUrl": "http://127.0.0.1:${provider_port}/v1", "apiKey": "fake-key", "api": "openai-completions",
  "models": [{"id": "local-test", "reasoning": false, "contextWindow": 100000, "maxTokens": 4096, "input": ["text"]},
             {"id": "local-reasoning", "reasoning": true, "contextWindow": 100000, "maxTokens": 4096, "input": ["text"]}]
}}}
EOF
cat > "$home_dir/.pi/agent/settings.json" <<'EOF'
{"packages": [], "extensions": [], "skills": [], "promptTemplates": [],
 "defaultProvider": "local", "defaultModel": "local-test",
 "retry": {"enabled": false}, "quietStartup": true, "defaultProjectTrust": "always"}
EOF
chmod 600 "$home_dir/.pi/agent/models.json" "$home_dir/.pi/agent/settings.json"

# Git redirects make the canonical upstream resolve to the offline fixture;
# the container's dead HTTP(S) proxy makes every other remote fail fast, so a
# checkout can only have come from the canonical fixture.
cat > "$home_dir/.gitconfig" <<EOF
[url "file:///fixture/canonical.git"]
	insteadOf = https://github.com/yiwenlu66/codoxear
	insteadOf = https://github.com/yiwenlu66/codoxear.git
[safe.directory]
	path = /fixture/canonical.git
EOF

# Deterministic streaming provider inside the container (same shape as
# scripts/docker_external_pi_verify.py, minus the slow-reply branch).
cat > "$home_dir/.verify-provider.py" <<EOF
import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROVIDER_LOG = "/home/tester/.verify-provider.log"

def log_provider(entry):
    with open(PROVIDER_LOG, "a", encoding="utf-8") as handle:
        handle.write(time.strftime("[%Y-%m-%dT%H:%M:%S] ") + entry + "\\n")

def content_text(content):
    # Pi may send a plain string or OpenAI content blocks; echo the raw text.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return str(content)

class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        log_provider("request " + self.path + " " + raw.decode("utf-8", "replace")[-20000:])
        payload = json.loads(raw or b"{}")
        users = [m for m in payload["messages"] if m["role"] == "user"]
        last = content_text(users[-1]["content"]) if users else ""
        reply = "LOCAL RESPONSE: " + last
        log_provider("reply " + reply[:500])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b'data: {"id":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\\n\\n')
            self.wfile.flush()
            for data in [
                {"id": "test", "choices": [{"index": 0, "delta": {"content": "LOCAL RESPONSE: " + last}}]},
                {"id": "test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}},
            ]:
                self.wfile.write(("data: " + json.dumps(data) + "\\n\\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\\n\\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

ThreadingHTTPServer(("127.0.0.1", ${provider_port}), Provider).serve_forever()
EOF

# ---------------------------------------------------------------------------
# Image and container.
# ---------------------------------------------------------------------------
mkdir -p "$root/source"
git -C "$repo_root" archive "$commit" | tar -x -C "$root/source"
npx --yes esbuild "$root/source/codoxear/static/app.js" --bundle --minify --format=esm \
  --outfile="$root/source/codoxear/static/dist/app.bundle.js" > "$artifacts/bundle.log" 2>&1
"${docker[@]}" build -f "$root/source/docker/verify.Dockerfile" -t "$image" "$root/source" \
  > "$artifacts/build.log" 2>&1 || fail 'image build'

"${docker[@]}" run -d --name "$container" --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester --env CODEX_WEB_PASSWORD="$password" --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" --env CODEX_WEB_DEFAULT_AGENT_BACKEND=pi \
  --env PYTHONPATH=/workspace --env PYTHONDONTWRITEBYTECODE=1 --env PI_OFFLINE=1 \
  --env HTTP_PROXY=http://127.0.0.1:9 --env HTTPS_PROXY=http://127.0.0.1:9 \
  --env http_proxy=http://127.0.0.1:9 --env https_proxy=http://127.0.0.1:9 \
  --env NO_PROXY=127.0.0.1,localhost --env no_proxy=127.0.0.1,localhost \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --mount "type=bind,src=$root/fixture,dst=/fixture,readonly" \
  "$image" bash -c "python3 /home/tester/.verify-provider.py >/home/tester/provider.log 2>&1 & exec python3 -m codoxear.server" \
  > "$artifacts/container-id.txt" || fail 'container start'

cookie="$root/cookies.txt"
for _ in $(seq 1 120); do
  if [[ "$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)" == 401 ]]; then
    break
  fi
  sleep 1
done
[[ "$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)" == 401 ]] || fail 'server never became reachable'

git_dirs_snapshot() {
  "${docker[@]}" exec "$container" sh -c 'find /home/tester /tmp -xdev -maxdepth 7 -type d -name .git 2>/dev/null | sort'
}

# Enable or disable the canonical-URL redirect inside the running container.
# "sabotage" removes it so the setup clone can only follow the dead proxy.
toggle_canonical_redirect() {
  "${docker[@]}" exec -i "$container" python3 - "$1" <<'PY'
import subprocess, sys
key = "url.file:///fixture/canonical.git.insteadOf"
if sys.argv[1] == "sabotage":
    subprocess.run(["git", "config", "--global", "--unset-all", key], check=False)
else:
    subprocess.run(["git", "config", "--global", "--replace-all", key, "https://github.com/yiwenlu66/codoxear"], check=True)
    subprocess.run(["git", "config", "--global", "--add", key, "https://github.com/yiwenlu66/codoxear.git"], check=True)
PY
}

# ---------------------------------------------------------------------------
# Stage 1: authentication. Unauthenticated reads and writes are both 401.
# ---------------------------------------------------------------------------
unauth_get="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/source-workspace")"
unauth_post="$(curl -sS -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:${port}/api/source-workspace")"
printf '{"unauthGet": "%s", "unauthPost": "%s"}\n' "$unauth_get" "$unauth_post" > "$artifacts/auth.json"

curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
  --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/login.json"
for _ in $(seq 1 120); do
  if [[ "$(curl -sS -b "$cookie" -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/sessions" 2>/dev/null || true)" == 200 ]]; then
    break
  fi
  sleep 1
done

# ---------------------------------------------------------------------------
# Stage 2: authenticated GET before any setup gesture. Absent shape, and no
# clone may happen as a side effect of GET.
# ---------------------------------------------------------------------------
git_dirs_snapshot > "$artifacts/git-dirs-baseline.txt"
curl_code="$(curl -sS -b "$cookie" -o "$artifacts/absent.json" -w '%{http_code}' "http://127.0.0.1:${port}/api/source-workspace")"
printf '{"authedGetStatus": "%s"}\n' "$curl_code" > "$artifacts/authed-get.json"
git_dirs_snapshot > "$artifacts/git-dirs-after-get.txt"
diff "$artifacts/git-dirs-baseline.txt" "$artifacts/git-dirs-after-get.txt" > /dev/null || fail 'GET /api/source-workspace cloned something'

# ---------------------------------------------------------------------------
# Browser session.
# ---------------------------------------------------------------------------
browser open "http://127.0.0.1:${port}/" > /dev/null 2>&1 || fail 'browser could not open app'
browser errors --clear --json > /dev/null 2>&1 || true
browser console --clear --json > /dev/null 2>&1 || true
evalv() { browser eval "$1" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["result"])'; }
wait_js() {
  local condition="$1"
  for _ in $(seq 1 240); do
    [[ "$(evalv "$condition")" != "True" ]] || return 0
    sleep 0.25
  done
  fail "browser condition timed out: $condition"
}

wait_js 'Boolean(document.getElementById("pw"))'
browser fill '#pw' "$password" > /dev/null
browser click '#loginBtn' > /dev/null
wait_js 'Boolean(document.getElementById("newBtn"))'
# One fetch chokepoint for the whole run: records every source-workspace call
# and every session-creating POST, and can delay source-workspace responses so
# close/typed-during-pending races become deterministic.
browser eval 'window.__nativeFetch = window.fetch.bind(window); window.__calls = []; window.__swDelay = false;
window.fetch = async function(url, opts) {
  const u = String(url); const m = ((opts && opts.method) || "GET").toUpperCase();
  if (u.includes("/api/source-workspace") || (m === "POST" && u.endsWith("/api/sessions"))) window.__calls.push({m, u, t: Date.now()});
  if (u.includes("/api/source-workspace") && window.__swDelay) { window.__swDelay = false; await new Promise(r => setTimeout(r, 2500)); }
  return window.__nativeFetch(url, opts);
}' > /dev/null

# ---------------------------------------------------------------------------
# Stage 3: absent source in the UI. The shortcut is a visible secondary in the
# working-directory row; the lazy GET offers setup and clones nothing.
# ---------------------------------------------------------------------------
browser click '#newBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser eval '(() => {
  const visible = (el) => Boolean(el) && getComputedStyle(el).display !== "none" && el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0;
  const btn = document.getElementById("newSessionSourceBtn");
  return visible(btn)
    && !btn.classList.contains("primary")
    && Boolean(btn.closest(".sourceWorkspaceHeading")?.querySelector("label[for=newSessionCwdInput]"))
    && document.documentElement.scrollWidth <= window.innerWidth;
})()' --json > "$artifacts/ui-shortcut-desktop.json"
[[ "$(evalv 'document.documentElement.scrollWidth <= window.innerWidth')" == "True" ]] || fail 'horizontal overflow with the new-session dialog open (desktop)'
browser click '#newSessionSourceBtn' > /dev/null
wait_js '!document.getElementById("newSessionSourceBtn").disabled && document.getElementById("newSessionSourceStatus").textContent.trim().length > 0'
browser eval '({cwd: document.getElementById("newSessionCwdInput").value, status: document.getElementById("newSessionSourceStatus").textContent,
  setupVisible: getComputedStyle(document.getElementById("newSessionSourceSetup")).display !== "none",
  contextHidden: getComputedStyle(document.getElementById("newSessionSourceContext")).display === "none",
  sourceCalls: window.__calls.filter(c => c.u.includes("/api/source-workspace")).length,
  sessionPosts: window.__calls.filter(c => c.u.endsWith("/api/sessions") && c.m === "POST").length})' --json > "$artifacts/ui-absent.json"
git_dirs_snapshot > "$artifacts/git-dirs-after-absent-click.txt"
diff "$artifacts/git-dirs-baseline.txt" "$artifacts/git-dirs-after-absent-click.txt" > /dev/null || fail 'absent-state source GET cloned something'
browser screenshot "$artifacts/absent-state.png" > /dev/null

# ---------------------------------------------------------------------------
# Stage 4: setup failure is explicit and visible. Removing the canonical
# redirect leaves only the dead-proxy network path, so the clone fails fast.
# ---------------------------------------------------------------------------
toggle_canonical_redirect sabotage
browser click '#newSessionSourceSetup' > /dev/null 2>&1 || fail 'setup affordance not clickable: backend route missing or absent state wrong (see ui-absent.json)'
wait_js '!document.getElementById("newSessionSourceSetup").disabled && document.getElementById("newSessionSourceStatus").textContent.trim().length > 0'
browser eval '({status: document.getElementById("newSessionSourceStatus").textContent,
  setupVisible: getComputedStyle(document.getElementById("newSessionSourceSetup")).display !== "none",
  cwd: document.getElementById("newSessionCwdInput").value,
  contextHidden: getComputedStyle(document.getElementById("newSessionSourceContext")).display === "none"})' --json > "$artifacts/setup-error.json"
git_dirs_snapshot > "$artifacts/git-dirs-after-error.txt"
diff "$artifacts/git-dirs-baseline.txt" "$artifacts/git-dirs-after-error.txt" > /dev/null || fail 'failed setup still left a clone behind'
browser screenshot "$artifacts/setup-error.png" > /dev/null

# ---------------------------------------------------------------------------
# Stage 5: explicit setup succeeds. Record the reported path and verify the
# checkout is a real, editable, canonically-rooted clone with the skills.
# ---------------------------------------------------------------------------
toggle_canonical_redirect restore
browser click '#newSessionSourceSetup' > /dev/null
wait_js '!document.getElementById("newSessionSourceSetup").disabled && document.getElementById("newSessionSourceSetup").style.display === "none"'
curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/source-workspace" > "$artifacts/available.json"
source_path="$(python3 -c '
import json, re, sys
data = json.load(open(sys.argv[1]))
path = data.get("path")
if data.get("available") is not True or not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", path or ""):
    raise SystemExit("setup did not return an available, usable path: " + json.dumps(data))
print(path)
' "$artifacts/available.json")" || fail 'setup did not produce a usable source path'
readonly source_path
browser eval "window.__sourcePath = '${source_path}'" > /dev/null
browser eval '({cwd: document.getElementById("newSessionCwdInput").value, path: window.__sourcePath,
  status: document.getElementById("newSessionSourceStatus").textContent,
  setupHidden: document.getElementById("newSessionSourceSetup").style.display === "none",
  contextVisible: getComputedStyle(document.getElementById("newSessionSourceContext")).display !== "none"})' --json > "$artifacts/setup-ok.json"

"${docker[@]}" exec -i "$container" python3 - "$source_path" "$fixture_sha" > "$artifacts/checkout.json" <<'PY'
import json, os, pathlib, subprocess, sys

path = pathlib.Path(sys.argv[1]); fixture_sha = sys.argv[2]
def git(*args):
    return subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True, check=True).stdout.strip()

clean = git("status", "--porcelain") == ""
head = git("rev-parse", "HEAD")
probe = subprocess.run(["git", "-C", str(path), "remote", "get-url", "origin"], text=True, capture_output=True)
origin = probe.stdout.strip() if probe.returncode == 0 else ""
inside = git("rev-parse", "--is-inside-work-tree") == "true"
app_dir = os.path.dirname(__import__("codoxear").__file__)
editable = False
try:
    (path / "codoxear_verify_edit.txt").write_text("editable proof\n")
    subprocess.run(["git", "-C", str(path), "add", "codoxear_verify_edit.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=verify", "-c", "user.email=verify@local",
                    "commit", "-m", "verify: editable checkout proof"], check=True, capture_output=True)
    editable = git("log", "-1", "--format=%s") == "verify: editable checkout proof"
except Exception:
    editable = False
skills = {}
for name in ("codoxear-report-issue", "codoxear-contribute-fix"):
    body = path / ".agents/skills" / name / "SKILL.md"
    link = path / ".claude/skills" / name
    skills[name] = {
        "bodyReadable": body.is_file() and len(body.read_text()) > 0,
        "claudeLink": link.is_symlink() and os.path.realpath(link).endswith(f".agents/skills/{name}") and (link / "SKILL.md").is_file(),
    }
print(json.dumps({
    "path": str(path), "clean": clean, "head": head, "headMatchesFixture": head == fixture_sha,
    "insideWorkTree": inside, "origin": origin,
    "originIsCanonical": origin in ("https://github.com/yiwenlu66/codoxear", "https://github.com/yiwenlu66/codoxear.git", "file:///fixture/canonical.git"),
    "notServingSnapshot": str(path) != "/workspace" and not str(path).startswith("/workspace/") and str(path) != app_dir,
    "writable": os.access(path, os.W_OK), "editableCommit": editable, "skills": skills,
}, indent=2))
PY

# ---------------------------------------------------------------------------
# Stage 6: POST is idempotent — same result, and it must not re-clone over
# existing state (an untracked marker file must survive the second POST).
# ---------------------------------------------------------------------------
"${docker[@]}" exec "$container" sh -c "printf 'idempotency marker\n' > '$source_path/.codoxear_idempotency_marker'"
post_code="$(curl -sS -b "$cookie" -o "$artifacts/setup-again.json" -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:${port}/api/source-workspace")"
printf '{"secondPostStatus": "%s"}\n' "$post_code" > "$artifacts/idempotent-status.json"
"${docker[@]}" exec "$container" sh -c "head -1 '$source_path/.codoxear_idempotency_marker' 2>/dev/null; git -C '$source_path' rev-parse HEAD" \
  > "$artifacts/idempotent-state.txt"

# ---------------------------------------------------------------------------
# Stage 7: opening the dialog alone must not GET, and with the source GET
# pending a browse-only open + dismissal of the cwd list must still let the
# source path land.
# ---------------------------------------------------------------------------
browser click '#newSessionCancelBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "none"'
browser click '#newBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser eval 'window.__callsBeforeOpen = window.__calls.length' > /dev/null
browser wait 400 > /dev/null
browser eval 'window.__callsAfterOpen = window.__calls.length' > /dev/null
browser eval 'window.__swDelay = true' > /dev/null
browser click '#newSessionSourceBtn' > /dev/null
browser wait 500 > /dev/null
browser click '#newSessionCwdInput' > /dev/null
browser eval 'document.getElementById("newSessionCwdMenu").classList.contains("open") && document.getElementById("newSessionCwdInput").getAttribute("aria-expanded") === "true"' --json > "$artifacts/browse-open.json"
browser press Escape > /dev/null
browser wait 300 > /dev/null
browser eval 'window.__callsBeforeLand = window.__calls.length' > /dev/null
browser eval 'window.__swDelay = false' > /dev/null
browser wait 3500 > /dev/null
browser eval '({cwd: document.getElementById("newSessionCwdInput").value, path: window.__sourcePath,
  status: document.getElementById("newSessionSourceStatus").textContent,
  contextVisible: getComputedStyle(document.getElementById("newSessionSourceContext")).display !== "none",
  openAutoGet: window.__callsAfterOpen === window.__callsBeforeOpen,
  callsStable: window.__calls.length === window.__callsBeforeLand})' --json > "$artifacts/browse-pending.json"

# ---------------------------------------------------------------------------
# Stage 8: a cwd the user types while the fetch is pending wins over the async
# landing, and the context help stays hidden.
# ---------------------------------------------------------------------------
browser click '#newSessionCancelBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "none"'
browser click '#newBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser eval 'window.__swDelay = true' > /dev/null
browser click '#newSessionSourceBtn' > /dev/null
browser wait 500 > /dev/null
browser fill '#newSessionCwdInput' '/tmp/typed-by-user' > /dev/null
browser eval 'window.__swDelay = false' > /dev/null
browser wait 3500 > /dev/null
browser eval '({cwd: document.getElementById("newSessionCwdInput").value,
  contextHidden: getComputedStyle(document.getElementById("newSessionSourceContext")).display === "none",
  status: document.getElementById("newSessionSourceStatus").textContent})' --json > "$artifacts/typed-pending.json"
browser click '#newSessionCancelBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "none"'

# ---------------------------------------------------------------------------
# Stage 9: the shortcut fills the editable checkout path, preserves
# backend/model, launches nothing, and its extended hit lane must not
# intercept the input below, the label beside it, or the header close button.
# ---------------------------------------------------------------------------
browser click '#newBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser click '#newSessionModelInput' > /dev/null
wait_js 'Boolean(document.getElementById("newSessionModelOption-0"))'
browser click '#newSessionModelOption-0' > /dev/null
browser wait 300 > /dev/null
browser eval '({backend: document.getElementById("newSessionBackendName").textContent, model: document.getElementById("newSessionModelInput").value})' \
  --json > "$artifacts/launch-selection-before.json"
browser eval '(() => {
  const at = (x, y) => document.elementFromPoint(x, y);
  const btn = document.getElementById("newSessionSourceBtn");
  const input = document.getElementById("newSessionCwdInput");
  const label = document.querySelector("label[for=newSessionCwdInput]");
  const closeBtn = document.getElementById("newSessionCloseBtn");
  const isBtn = (hit) => Boolean(hit) && (hit === btn || btn.contains(hit));
  const center = (el) => { const r = el.getBoundingClientRect(); return at(r.left + r.width / 2, r.top + r.height / 2); };
  const tag = (hit) => !hit ? "none" : hit === input ? "input" : hit === closeBtn ? "close" : (hit === label || (label && label.contains(hit))) ? "label" : isBtn(hit) ? "button" : "other";
  const rect = btn.getBoundingClientRect();
  const laneTop = rect.bottom + 3 - 44; const laneBottom = rect.bottom + 3;
  const scroller = btn.closest(".formBody");
  const viewport = scroller ? scroller.getBoundingClientRect() : {top: 0, bottom: window.innerHeight};
  const samples = [];
  for (const f of [0.15, 0.4, 0.6, 0.85]) {
    const y = laneTop + f * (laneBottom - laneTop);
    const hit = at(rect.left + rect.width / 2, y);
    samples.push({f, insideScrollport: y >= viewport.top && y <= viewport.bottom, hitIsButton: isBtn(hit), hitTag: tag(hit)});
  }
  const effective = samples.filter((s) => s.insideScrollport);
  return {
    laneSamples: samples,
    laneEffectiveCount: effective.length,
    laneExtensionHit: effective.some((s) => s.hitIsButton && (laneTop + s.f * (laneBottom - laneTop)) < rect.top),
    laneNoIntercepts: samples.every((s) => ["input", "close", "label"].indexOf(s.hitTag) === -1),
    inputNotIntercepted: !isBtn(center(input)),
    labelNotIntercepted: !isBtn(center(label)),
    closeNotIntercepted: !isBtn(center(closeBtn)),
  };
})()' --json > "$artifacts/hitlane-desktop.json"
browser click '#newSessionSourceBtn' > /dev/null
wait_js '!document.getElementById("newSessionSourceBtn").disabled && document.getElementById("newSessionCwdInput").value === window.__sourcePath'
browser eval '({cwd: document.getElementById("newSessionCwdInput").value, path: window.__sourcePath,
  backend: document.getElementById("newSessionBackendName").textContent, model: document.getElementById("newSessionModelInput").value,
  contextVisible: getComputedStyle(document.getElementById("newSessionSourceContext")).display !== "none",
  sessionPosts: window.__calls.filter(c => c.u.endsWith("/api/sessions") && c.m === "POST").length,
  selected: localStorage.getItem("codexweb.selected")})' --json > "$artifacts/shortcut-fill.json"
sidecars="$("${docker[@]}" exec "$container" sh -c 'ls "$HOME/.local/share/codoxear/socks" 2>/dev/null | wc -l')"
printf '{"sidecarFiles": "%s"}\n' "$sidecars" > "$artifacts/sidecars-before-launch.json"
curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/sessions-before-launch.json"

# ---------------------------------------------------------------------------
# Stage 10: editing the cwd by hand removes the contextual source help.
# ---------------------------------------------------------------------------
browser fill '#newSessionCwdInput' '/tmp' > /dev/null
browser wait 300 > /dev/null
browser eval '({cwd: document.getElementById("newSessionCwdInput").value,
  contextHidden: getComputedStyle(document.getElementById("newSessionSourceContext")).display === "none"})' \
  --json > "$artifacts/manual-edit.json"

# ---------------------------------------------------------------------------
# Stage 11: the Help entry closes Help and opens the same dialog prefilled.
# ---------------------------------------------------------------------------
browser click '#newSessionCancelBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "none"'
browser click '#helpBtnSide' > /dev/null
wait_js 'getComputedStyle(document.getElementById("helpViewer")).display === "flex"'
browser click '#helpSourceBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser eval '({helpHidden: getComputedStyle(document.getElementById("helpViewer")).display === "none" && getComputedStyle(document.getElementById("helpBackdrop")).display === "none",
  newSessionOpen: getComputedStyle(document.getElementById("newSessionViewer")).display === "flex" && getComputedStyle(document.getElementById("newSessionBackdrop")).display === "block"})' \
  --json > "$artifacts/help-entry-modal.json"
wait_js '!document.getElementById("newSessionSourceBtn").disabled && document.getElementById("newSessionCwdInput").value === window.__sourcePath'
browser eval '({cwd: document.getElementById("newSessionCwdInput").value,
  contextVisible: getComputedStyle(document.getElementById("newSessionSourceContext")).display !== "none"})' --json > "$artifacts/help-entry-fill.json"
browser screenshot "$artifacts/help-prefilled.png" > /dev/null

# ---------------------------------------------------------------------------
# Stage 12: launch a real session in the source workspace and prove it runs
# there: sidecar cwd, backend process cwd, deterministic assistant reply.
# ---------------------------------------------------------------------------
browser eval 'document.getElementById("newSessionTmuxToggle").checked = false' > /dev/null
browser click '#newSessionStartBtn' > /dev/null
wait_js 'Boolean(localStorage.getItem("codexweb.selected")) && !document.getElementById("sendBtn").disabled'
source_sid="$(evalv 'localStorage.getItem("codexweb.selected")')"
[[ "$source_sid" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unexpected session id: $source_sid"
browser eval "window.__sourceSid = '${source_sid}'" > /dev/null
browser wait 3500 > /dev/null
browser fill '#msg' 'Reply exactly SOURCE_WORKSPACE_OK.' > /dev/null
browser click '#sendBtn' > /dev/null
wait_js 'document.getElementById("chatInner").textContent.includes("LOCAL RESPONSE")'
browser eval '({helpHidden: getComputedStyle(document.getElementById("helpViewer")).display === "none" && getComputedStyle(document.getElementById("helpBackdrop")).display === "none",
  composerUsable: !document.getElementById("msg").disabled && !document.getElementById("sendBtn").disabled,
  newSessionClosed: getComputedStyle(document.getElementById("newSessionViewer")).display === "none"})' --json > "$artifacts/help-entry-after-start.json"
curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions/${source_sid}/messages/tail?limit=100" > "$artifacts/source-session-tail.json"
printf '{"sourceSid": "%s"}\n' "$source_sid" > "$artifacts/source-session.json"
browser screenshot "$artifacts/source-session.png" > /dev/null

# ---------------------------------------------------------------------------
# Stage 13: an ordinary new session is unaffected.
# ---------------------------------------------------------------------------
browser click '#newBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser eval 'document.getElementById("newSessionTmuxToggle").checked = false' > /dev/null
browser fill '#newSessionCwdInput' '/tmp' > /dev/null
browser wait 300 > /dev/null
browser eval '({contextHidden: getComputedStyle(document.getElementById("newSessionSourceContext")).display === "none"})' \
  --json > "$artifacts/normal-dialog.json"
browser click '#newSessionStartBtn' > /dev/null
wait_js 'Boolean(localStorage.getItem("codexweb.selected")) && localStorage.getItem("codexweb.selected") !== window.__sourceSid && !document.getElementById("sendBtn").disabled'
browser wait 3500 > /dev/null
browser fill '#msg' 'Reply exactly NORMAL_SESSION_OK.' > /dev/null
browser click '#sendBtn' > /dev/null
wait_js 'document.getElementById("chatInner").textContent.includes("LOCAL RESPONSE")'
normal_sid="$(evalv 'localStorage.getItem("codexweb.selected")')"
printf '{"normalSid": "%s"}\n' "$normal_sid" > "$artifacts/normal-session.json"

# Broker/filesystem ground truth for both launches.
"${docker[@]}" exec -i "$container" python3 - > "$artifacts/brokers.json" <<'PY'
import json, os, pathlib
from codoxear.control_socket import call_control_socket
rows = []
for meta_path in sorted(pathlib.Path.home().joinpath(".local/share/codoxear/socks").glob("*.json")):
    meta = json.loads(meta_path.read_text())
    agent_pid = int(meta.get("codex_pid") or 0)
    proc_cwd = None
    if agent_pid > 0 and os.path.exists(f"/proc/{agent_pid}/cwd"):
        proc_cwd = os.readlink(f"/proc/{agent_pid}/cwd")
    sock = meta_path.with_suffix(".sock")
    rows.append({
        "session_id": meta_path.stem, "cwd": meta.get("cwd"), "agent_backend": meta.get("agent_backend"),
        "proc_cwd": proc_cwd,
        "state": call_control_socket(sock, {"cmd": "state"}),
        "tail": str(call_control_socket(sock, {"cmd": "tail"}).get("tail", ""))[-4000:],
    })
print(json.dumps({"brokers": rows}, indent=2))
PY

# ---------------------------------------------------------------------------
# Stage 14: mobile viewport. Shortcut visible, no horizontal overflow, hit
# lane still does not intercept the input.
# ---------------------------------------------------------------------------
browser set viewport 390 844 > /dev/null
browser click '#toggleSidebarBtn' > /dev/null
browser click '#newBtn' > /dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display === "flex"'
browser eval '(() => {
  const visible = (el) => Boolean(el) && getComputedStyle(el).display !== "none" && el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0;
  const btn = document.getElementById("newSessionSourceBtn");
  const input = document.getElementById("newSessionCwdInput");
  const rect = input.getBoundingClientRect();
  const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
  return {shortcutVisible: visible(btn), notPrimary: !btn.classList.contains("primary"),
    overflow: document.documentElement.scrollWidth > window.innerWidth,
    inputNotIntercepted: Boolean(hit) && !(hit === btn || btn.contains(hit))};
})()' --json > "$artifacts/mobile.json"
browser screenshot "$artifacts/mobile-dialog.png" > /dev/null

browser errors --json > "$artifacts/browser-errors.json"
browser console --json > "$artifacts/browser-console.json"
"${docker[@]}" exec "$container" sh -c 'cat "$HOME/.verify-provider.log" 2>/dev/null; true' > "$artifacts/provider-requests.log" 2>&1 || true

# ---------------------------------------------------------------------------
# Verdict: every stage's captured evidence is asserted in one place.
# ---------------------------------------------------------------------------
if python3 - "$artifacts" "$commit" "$source_path" "$fixture_sha" <<'PY'
import json, pathlib, sys

artifacts, commit, source_path, fixture_sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
problems = []

def read_json(name):
    return json.loads((artifacts / name).read_text())

def browser_result(name):
    payload = read_json(name)
    if not isinstance(payload, dict) or payload.get("success") is not True or not isinstance(payload.get("data"), dict):
        problems.append(f"{name}: agent-browser returned an unsuccessful response")
        return {}
    return payload["data"].get("result") or {}

def expect(condition, message):
    if not condition:
        problems.append(message)

auth = read_json("auth.json")
expect(auth.get("unauthGet") == "401", f"unauthenticated GET status is {auth.get('unauthGet')}, not 401")
expect(auth.get("unauthPost") == "401", f"unauthenticated POST status is {auth.get('unauthPost')}, not 401")
authed = read_json("authed-get.json")
expect(authed.get("authedGetStatus") == "200", f"authenticated GET status is {authed.get('authedGetStatus')}, not 200")
absent = read_json("absent.json")
expect(absent.get("available") is False, f"pre-setup GET reports available={absent.get('available')!r}")
expect(absent.get("path") in (None, ""), f"pre-setup GET reports a path {absent.get('path')!r}")
expect(absent.get("can_setup") is True, f"pre-setup GET reports can_setup={absent.get('can_setup')!r}")
expect(isinstance(absent.get("reason"), str) and absent.get("reason", "").strip(), "pre-setup GET carries no reason")

shortcut = browser_result("ui-shortcut-desktop.json")
expect(shortcut is True, "desktop: shortcut is not a visible secondary in the working-directory row")
absent_ui = browser_result("ui-absent.json")
expect(absent_ui.get("cwd", "x") == "", f"absent-state GET filled cwd with {absent_ui.get('cwd')!r}")
expect(absent_ui.get("setupVisible") is True, "absent state does not offer the explicit setup affordance")
expect(absent_ui.get("contextHidden") is True, "absent state shows the contextual source help")
expect(absent_ui.get("sessionPosts") == 0, "absent-state source GET created a session")
expect(absent_ui.get("sourceCalls") == 1, f"expected exactly one lazy source GET, saw {absent_ui.get('sourceCalls')}")

setup_error = browser_result("setup-error.json")
expect(isinstance(setup_error.get("status"), str) and setup_error["status"].strip(), "failed setup shows no visible error")
expect(setup_error.get("setupVisible") is True, "failed setup lost the setup affordance")
expect(setup_error.get("cwd", "x") == "", "failed setup filled the cwd")
expect(setup_error.get("contextHidden") is True, "failed setup showed the contextual help")

setup_ok = browser_result("setup-ok.json")
expect(setup_ok.get("cwd") == source_path, f"successful setup left cwd {setup_ok.get('cwd')!r}, wanted {source_path}")
expect(setup_ok.get("setupHidden") is True, "successful setup left the setup affordance visible")
expect(setup_ok.get("contextVisible") is True, "successful setup did not show the contextual help")
available = read_json("available.json")
expect(available.get("available") is True and available.get("path") == source_path, "post-setup GET disagrees with the setup result")

checkout = read_json("checkout.json")
expect(checkout.get("insideWorkTree") is True, f"{checkout.get('path')}: not a git work tree")
expect(checkout.get("clean") is True, "fresh setup checkout is not clean")
expect(checkout.get("headMatchesFixture") is True, f"checkout HEAD {checkout.get('head')} is not the canonical fixture {fixture_sha}")
expect(checkout.get("originIsCanonical") is True, f"checkout origin {checkout.get('origin')!r} is not the canonical upstream")
expect(checkout.get("notServingSnapshot") is True, "setup produced the serving snapshot/package directory, not an editable checkout")
expect(checkout.get("writable") is True and checkout.get("editableCommit") is True, "setup checkout is not an editable, committable clone")
for name, facts in (checkout.get("skills") or {}).items():
    expect(facts.get("bodyReadable") is True, f"skill {name}: canonical SKILL.md missing from the setup checkout")
    expect(facts.get("claudeLink") is True, f"skill {name}: .claude/skills symlink missing or unresolved")

idempotent_status = read_json("idempotent-status.json")
expect(idempotent_status.get("secondPostStatus") == "200", f"second setup POST returned {idempotent_status.get('secondPostStatus')}")
idempotent_again = read_json("setup-again.json")
expect(idempotent_again.get("available") is True and idempotent_again.get("path") == source_path, "second setup POST returned a different workspace")
marker = (artifacts / "idempotent-state.txt").read_text().splitlines()
expect(bool(marker) and marker[0].strip() == "idempotency marker", "second setup POST destroyed existing checkout state (marker lost)")
expect(len(marker) > 1 and marker[1].strip() == fixture_sha, "second setup POST moved the checkout HEAD")

close_inv = browser_result("browse-pending.json")
expect(close_inv.get("openAutoGet") is True, "opening the dialog fired an automatic source GET")
expect(close_inv.get("cwd") == source_path, f"browse-only interaction during a pending GET left cwd {close_inv.get('cwd')!r}, wanted {source_path}")
expect(close_inv.get("contextVisible") is True, "post-browse source fill does not show the contextual help")
expect(close_inv.get("status", "x") == "", "browse-only interaction left status text behind")
expect(close_inv.get("callsStable") is True, "browse interaction or the pending landing fired extra source GETs")
browse_open = browser_result("browse-open.json")
expect(browse_open is True, "the cwd browse list did not open while the source GET was pending")

typed = browser_result("typed-pending.json")
expect(typed.get("cwd") == "/tmp/typed-by-user", f"pending fetch overwrote the typed cwd with {typed.get('cwd')!r}")
expect(typed.get("contextHidden") is True, "typed non-source cwd still shows the contextual help")

before = browser_result("launch-selection-before.json")
after = browser_result("shortcut-fill.json")
expect(after.get("cwd") == source_path, f"shortcut left cwd {after.get('cwd')!r}")
expect(str(before.get("model") or "").strip() != "", "model pre-selection for the preservation check failed")
expect(after.get("backend") == before.get("backend") and after.get("model") == before.get("model"), "shortcut did not preserve the chosen backend/model")
expect(after.get("contextVisible") is True, "source cwd does not show the contextual help")
expect(after.get("sessionPosts") == 0, "shortcut click launched a session")
expect(after.get("selected") in (None, ""), "shortcut click selected a session")
sidecars = read_json("sidecars-before-launch.json")
expect(str(sidecars.get("sidecarFiles")) == "0", f"sidecars appeared before any launch: {sidecars}")
sessions_before = read_json("sessions-before-launch.json")
expect(not (sessions_before.get("sessions") or []), "sessions existed before any launch")
hitlane = browser_result("hitlane-desktop.json")
expect(hitlane.get("laneEffectiveCount", 0) >= 1, "hit-lane samples found no visible scrollport overlap")
expect(hitlane.get("laneExtensionHit") is True, "the 44px hit lane does not reach above the link inside the scrollport")
expect(hitlane.get("laneNoIntercepts") is True, f"hit-lane samples intercepted a neighbor: {hitlane.get('laneSamples')}")
expect(hitlane.get("inputNotIntercepted") is True, "source shortcut hit lane intercepts the cwd input")
expect(hitlane.get("labelNotIntercepted") is True, "source shortcut hit lane intercepts the directory label")
expect(hitlane.get("closeNotIntercepted") is True, "source shortcut hit lane intercepts the header close button")
manual = browser_result("manual-edit.json")
expect(manual.get("cwd") == "/tmp" and manual.get("contextHidden") is True, "manual cwd edit did not remove the contextual help")

help_modal = browser_result("help-entry-modal.json")
expect(help_modal.get("helpHidden") is True, "Help entry left the Help modal open")
expect(help_modal.get("newSessionOpen") is True, "Help entry did not open the New session dialog")
help_fill = browser_result("help-entry-fill.json")
expect(help_fill.get("cwd") == source_path, f"Help entry dialog cwd is {help_fill.get('cwd')!r}")
expect(help_fill.get("contextVisible") is True, "Help-prefilled dialog does not show the contextual help")
help_after_start = browser_result("help-entry-after-start.json")
expect(help_after_start.get("helpHidden") is True, "Help reappeared after starting the source session")
expect(help_after_start.get("composerUsable") is True, "composer is inert after starting from the Help handoff")
expect(help_after_start.get("newSessionClosed") is True, "New session dialog stayed open after Start")

source_session = read_json("source-session.json")
source_sid = source_session.get("sourceSid")
tail = read_json("source-session-tail.json")
events = tail.get("events") or []
expect(any(e.get("role") == "user" and "SOURCE_WORKSPACE_OK" in str(e.get("text") or "") for e in events),
       "source-workspace session transcript lost the user prompt")
expect(any(e.get("role") == "assistant" and "LOCAL RESPONSE: Reply exactly SOURCE_WORKSPACE_OK." in str(e.get("text") or "") for e in events),
       "source-workspace session produced no deterministic assistant reply")

brokers = read_json("brokers.json").get("brokers") or []
by_sid = {row.get("session_id"): row for row in brokers}
expect(source_sid in by_sid, "source-workspace session has no broker sidecar")
source_row = by_sid.get(source_sid) or {}
expect(source_row.get("cwd") == source_path, f"source session sidecar cwd is {source_row.get('cwd')!r}")
expect(source_row.get("proc_cwd") == source_path, f"source session backend process cwd is {source_row.get('proc_cwd')!r}")
expect(source_row.get("agent_backend") == "pi", f"source session backend is {source_row.get('agent_backend')!r}")
expect("SOURCE_WORKSPACE_OK" in str(source_row.get("tail") or ""), "source session broker never received the prompt")

normal = read_json("normal-session.json")
normal_sid = normal.get("normalSid")
expect(normal_sid and normal_sid != source_sid, "ordinary launch did not create a second session")
normal_row = by_sid.get(normal_sid) or {}
expect(normal_row.get("cwd") == "/tmp", f"ordinary session cwd is {normal_row.get('cwd')!r}, wanted /tmp")
expect(normal_row.get("agent_backend") == "pi", f"ordinary session backend is {normal_row.get('agent_backend')!r}")
expect("NORMAL_SESSION_OK" in str(normal_row.get("tail") or ""), "ordinary session broker never received the prompt")
normal_dialog = browser_result("normal-dialog.json")
expect(normal_dialog.get("contextHidden") is True, "ordinary non-source cwd shows the contextual help")

mobile = browser_result("mobile.json")
expect(mobile.get("shortcutVisible") is True and mobile.get("notPrimary") is True, "mobile: shortcut is not a visible secondary")
expect(mobile.get("overflow") is False, "mobile: horizontal overflow with the dialog open")
expect(mobile.get("inputNotIntercepted") is True, "mobile: shortcut hit lane intercepts the cwd input")

errors_payload = read_json("browser-errors.json")
page_errors = errors_payload.get("data", {}).get("errors") if isinstance(errors_payload, dict) else None
expect(page_errors == [], f"browser page errors: {page_errors!r}")

report = {
    "commit": commit, "sourcePath": source_path, "fixtureHead": fixture_sha,
    "sourceSid": source_sid, "normalSid": normal_sid,
    "skillsDiscovery": "validated by the docs-refinement harness; this run asserts checkout skill files only",
    "pass": not problems, "problems": problems,
}
(artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n")
for problem in problems:
    print(f"CHECK FAIL: {problem}")
if problems:
    raise SystemExit(1)
print("PASS: source workspace auth, absent-state, explicit setup, idempotency, editable canonical checkout,")
print("      shortcut fill + preservation + hit lane, help entry, real source-cwd session, ordinary session, mobile")
PY
then
  echo "Artifacts: $artifacts"
else
  status=$?
  echo "FAIL: verification checks failed (artifacts: $artifacts)" >&2
  exit "$status"
fi
