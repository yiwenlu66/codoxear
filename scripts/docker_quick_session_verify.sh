#!/usr/bin/env bash
# Commit-scoped quick-launch acceptance using a real Pi broker and a real browser.
# Only container HOME receives credentials/runtime; nothing touches the deployment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="$(git -C "$repo_root" rev-parse --verify "${1:-HEAD}^{commit}")"
short="${commit:0:12}"
port="${CODOXEAR_QUICK_SESSION_PORT:-19679}"
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && ((port <= 65535)) && [[ "$port" != 8743 ]] || { echo 'invalid or live port' >&2; exit 2; }
[[ -z "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]] || { echo 'refusing app-dir override' >&2; exit 2; }
root="$(mktemp -d /tmp/codoxear-quick-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-quick-verify.XXXXXX)"
container="codoxear-quick-${short}-$$"
browser_session="codoxear-quick-${short}-$$"
image="${CODOXEAR_QUICK_SESSION_IMAGE:-codoxear-verify:${short}}"
password=docker-quick-session-password
readonly -a docker=(sudo docker)
browser() { AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"; }
cleanup() { browser close >/dev/null 2>&1 || true; "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true; /bin/rm -rf "$root"; }
trap cleanup EXIT
fail() { echo "FAIL: $* (artifacts: $artifacts)" >&2; "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true; exit 1; }
mkdir -p "$root/source" "$root/home/.pi/agent"
umask 077
for name in models.json models-store.json auth.json settings.json trust.json; do
  [[ ! -r "$HOME/.pi/agent/$name" ]] || install -m 600 "$HOME/.pi/agent/$name" "$root/home/.pi/agent/$name"
done
[[ -r "$root/home/.pi/agent/models.json" ]] || fail 'Pi model config required'
# Pi allows API keys to come from shell reads. Copy only literal `!cat` file
# dependencies under its agent directory; never execute configured commands.
python3 - "$HOME/.pi/agent" "$root/home/.pi/agent" <<'PY'
import json, pathlib, re, shutil, sys
source, target=map(pathlib.Path, sys.argv[1:])
models=json.loads((source/'models.json').read_text())
for provider in models.get('providers', {}).values():
    value=provider.get('apiKey', '')
    match=re.fullmatch(r'!cat ~/.pi/agent/([A-Za-z0-9._/-]+)', value) if isinstance(value,str) else None
    if not match: continue
    src=(source/match[1]).resolve()
    if not src.is_relative_to(source.resolve()) or not src.is_file():
        raise SystemExit('Configured Pi API-key file dependency unavailable')
    dst=target/match[1]; dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst); dst.chmod(0o600)
PY
# Keep provider/model defaults, not the user's package auto-install workflow.
# Package clones can hold Pi at startup before it is able to receive a prompt.
python3 - "$root/home/.pi/agent/settings.json" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); settings=json.loads(p.read_text()) if p.exists() else {}
for key in ('packages','extensions','skills','promptTemplates'):
    settings[key]=[]
settings['defaultProjectTrust']='always'
p.write_text(json.dumps(settings))
PY
git -C "$repo_root" archive "$commit" | tar -x -C "$root/source"
npx --yes esbuild "$root/source/codoxear/static/app.js" --bundle --minify --format=esm --outfile="$root/source/codoxear/static/dist/app.bundle.js" > "$artifacts/build.log" 2>&1
"${docker[@]}" build -f "$root/source/docker/verify.Dockerfile" -t "$image" "$root/source" >> "$artifacts/build.log" 2>&1 || fail 'image build'
"${docker[@]}" run -d --name "$container" --publish "127.0.0.1:$port:$port" \
  --env HOME=/home/tester --env CODEX_WEB_PASSWORD="$password" --env CODEX_WEB_HOST=0.0.0.0 --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DEFAULT_AGENT_BACKEND=pi --env PYTHONPATH=/workspace --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$root/home,dst=/home/tester" "$image" > "$artifacts/container-id.txt"
for _ in $(seq 1 120); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/api/me" || true)" != 401 ]] || break
  sleep 1
done
"${docker[@]}" exec "$container" python3 -m pytest -q -p no:cacheprovider \
  tests/test_quick_session.py tests/test_quick_session_composer.py tests/test_composer_controls.py \
  tests/test_composer_height_mode.py tests/test_escape_modal_policy_source.py \
  tests/test_new_session_model_options_source.py tests/test_frontend_wiring.py tests/test_check_wiring_guard.py \
  > "$artifacts/pytest.txt" 2>&1 || fail 'VM regressions'
evalv() { browser eval "$1" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["result"])'; }
wait_js() {
  local condition="$1"
  for _ in $(seq 1 120); do
    [[ "$(evalv "$condition")" != True ]] || return 0
    sleep 0.25
  done
  fail "browser condition timed out: $condition"
}
browser open "http://127.0.0.1:$port/" >/dev/null
browser fill '#pw' "$password" >/dev/null
browser click '#loginBtn' >/dev/null
wait_js 'Boolean(document.getElementById("quickNewBtn"))'
browser screenshot "$artifacts/empty-state.png" >/dev/null
# The full dialog supplies the record. Ordinary opens must still use defaults.
browser click '#newBtn' >/dev/null
browser fill '#newSessionCwdInput' /tmp >/dev/null
browser eval 'document.getElementById("newSessionTmuxToggle").checked=false; document.getElementById("newSessionReasoningBtn").click(); const choices=[...document.querySelectorAll("#newSessionReasoningMenu button")]; (choices.find(n=>n.textContent.trim()==="low")||choices[0]).click()' >/dev/null
browser click '#newSessionStartBtn' >/dev/null
wait_js 'Boolean(localStorage.getItem("codoxear.lastLaunch")) && Boolean(localStorage.getItem("codexweb.selected"))'
browser eval 'JSON.parse(localStorage.getItem("codoxear.lastLaunch"))' --json > "$artifacts/last-launch.json"
browser screenshot "$artifacts/sidebar-selected.png" >/dev/null
browser eval 'window.__quickDialogSession=localStorage.getItem("codexweb.selected")' >/dev/null
# Delay the next actual launch request so typing during boot is deterministic.
# Capture its unmodified launch body; the real server/broker still execute it.
browser eval 'window.__quickNativeFetch=window.fetch.bind(window); window.__quickDelay=true; window.__quickBodies=[]; window.__sendAcks=[]; window.fetch=async function(url,opts){if(String(url).endsWith("/api/sessions")&&opts&&opts.method==="POST"){const body=JSON.parse(opts.body); window.__quickBodies.push(body); if(window.__quickFail){window.__quickFail=false;opts={...opts,body:JSON.stringify({...body,reasoning_effort:"__verify_invalid__"})}} if(window.__quickDelay){window.__quickDelay=false;await new Promise(r=>setTimeout(r,5000))}}const response=await window.__quickNativeFetch(url,opts);if(String(url).endsWith("/send"))window.__sendAcks.push({url:String(url),status:response.status,body:await response.clone().json()});return response}' >/dev/null
browser click '#quickNewBtn' >/dev/null
wait_js 'document.querySelector(".chatEmptyCopy").textContent.startsWith("Starting ") && !document.getElementById("msg").disabled'
browser fill '#msg' 'Reply exactly QUICK_SESSION_OK.' >/dev/null
browser eval '({focused:document.activeElement.id,sendDisabled:document.getElementById("sendBtn").disabled,label:document.getElementById("sendBtn").title,copy:document.querySelector(".chatEmptyCopy").textContent,selected:localStorage.getItem("codexweb.selected")})' --json > "$artifacts/starting.json"
browser screenshot "$artifacts/starting.png" >/dev/null
wait_js 'Boolean(localStorage.getItem("codexweb.selected")) && !document.getElementById("sendBtn").disabled'
wait_js 'document.getElementById("msg").value === "Reply exactly QUICK_SESSION_OK."'
browser eval '(async()=>{const data=await (await window.__quickNativeFetch("/api/sessions")).json();const selected=localStorage.getItem("codexweb.selected");return {session:data.sessions.find(s=>s.session_id===selected),record:JSON.parse(localStorage.getItem("codoxear.lastLaunch")),body:window.__quickBodies[0],draft:document.getElementById("msg").value}})()' --json > "$artifacts/selected.json"
# Give the real TUI time to register readiness, then exercise the actual Send.
browser wait 3500 >/dev/null
browser click '#sendBtn' >/dev/null
wait_js 'document.querySelector("#chatInner").textContent.includes("QUICK_SESSION_OK") && document.getElementById("msg").value === ""'
# Provider success is separate from send delivery. The real PTY evidence below
# proves both launch paths received the prompt even when provider auth/network
# prevents Pi's first assistant response (and therefore defers log persistence).
browser wait 6000 >/dev/null
browser eval '(async()=>{const sid=localStorage.getItem("codexweb.selected");const r=await (await window.__quickNativeFetch(`/api/sessions/${sid}/messages/tail?limit=100`)).json();return {acks:window.__sendAcks,echo:document.querySelector("#chatInner").textContent.includes("QUICK_SESSION_OK"),draft:document.getElementById("msg").value,events:r.events}})()' --json > "$artifacts/delivered.json"
# Compare with the ordinary dialog launch to distinguish startup/provider issues
# from the quick-launch handoff. Both use the same real browser Send path.
browser eval 'document.querySelector(`#sessions .session[data-session-id="${window.__quickDialogSession}"]`).click()' >/dev/null
wait_js 'localStorage.getItem("codexweb.selected") === window.__quickDialogSession && !document.getElementById("sendBtn").disabled'
browser fill '#msg' 'Reply exactly QUICK_SESSION_OK.' >/dev/null
browser click '#sendBtn' >/dev/null
browser wait 6000 >/dev/null
browser eval '({acks:window.__sendAcks,draft:document.getElementById("msg").value,echo:document.querySelector("#chatInner").textContent.includes("QUICK_SESSION_OK")})' --json > "$artifacts/dialog-send.json"
"${docker[@]}" exec -i "$container" python3 - > "$artifacts/broker-evidence.json" <<'PY'
import json, pathlib, re
from codoxear.control_socket import call_control_socket
from codoxear.launch_ledger import redact_launch_failure_text
rows=[]
for path in pathlib.Path.home().joinpath('.local/share/codoxear/socks').glob('*.json'):
    meta=json.loads(path.read_text())
    tail=call_control_socket(path.with_suffix('.sock'), {'cmd':'tail'}).get('tail','')
    tail=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', tail)
    rows.append({'cwd':meta.get('cwd'),'log_path':meta.get('log_path'),'session_id':path.stem,
                 'state':call_control_socket(path.with_suffix('.sock'), {'cmd':'state'}),
                 'tail':redact_launch_failure_text(tail[-14000:])})
print(json.dumps(rows,indent=2))
PY
# A real server validation error must preserve the composer and show a preset.
browser eval 'window.__quickFail=true;window.__quickDelay=true' >/dev/null
browser click '#quickNewBtn' >/dev/null
browser fill '#msg' 'Preserve this failed-start draft.' >/dev/null
wait_js 'getComputedStyle(document.getElementById("newSessionViewer")).display !== "none"'
browser eval '({draft:document.getElementById("msg").value,cwd:document.getElementById("newSessionCwdInput").value,model:document.getElementById("newSessionModelInput").value,effort:document.getElementById("newSessionReasoningBtn").textContent,status:document.getElementById("newSessionStatus").textContent,record:JSON.parse(localStorage.getItem("codoxear.lastLaunch"))})' --json > "$artifacts/failure.json"
browser screenshot "$artifacts/failure.png" >/dev/null
# Responsive shell: a quick launch closes the narrow drawer and focuses input.
browser click '#newSessionCancelBtn' >/dev/null
browser set viewport 390 844 >/dev/null
browser click '#toggleSidebarBtn' >/dev/null
browser eval 'window.__quickDelay=true;window.__quickFail=true' >/dev/null
browser click '#quickNewBtn' >/dev/null
browser eval '({width:innerWidth,focused:document.activeElement.id,sidebarOpen:document.body.classList.contains("sidebar-open"),overflow:document.documentElement.scrollWidth>innerWidth})' --json > "$artifacts/mobile.json"
browser screenshot "$artifacts/mobile-starting.png" >/dev/null
browser errors --json > "$artifacts/browser-errors.json"
python3 - "$artifacts" "$commit" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
def read(n): return json.loads((p/(n+'.json')).read_text())['data']['result']
a=read('selected'); r=a['record']; s=a['session']; b=a['body']
assert s['cwd']=='/home/tester', s
assert s['agent_backend']==r['backend']=='pi', s
assert s['model']==r['model'] and s['reasoning_effort']==r['reasoningEffort'], (s,r)
assert b['cwd']=='~' and b['agent_backend']==r['backend'] and b['model']==r['model'], b
assert read('starting')['sendDisabled'] and read('starting')['focused']=='msg'
delivery=read('delivered'); comparison=read('dialog-send')
assert delivery['echo'] and delivery['draft']=='' and len(delivery['acks'])==1, delivery
assert delivery['acks'][0]['status']==200 and delivery['acks'][0]['body'].get('queued') is False, delivery
assistant_reply=any(e.get('role')=='assistant' and 'QUICK_SESSION_OK' in e.get('text','') for e in delivery['events'])
brokers=json.loads((p/'broker-evidence.json').read_text())
assert len(brokers)==2 and all('Reply exactly QUICK_SESSION_OK.' in row['tail'] for row in brokers), brokers
provider_errors=[line.strip() for row in brokers for line in row['tail'].splitlines() if 'Error:' in line]
assert len(comparison['acks'])==2 and comparison['acks'][-1]['status']==200, comparison
f=read('failure'); assert f['draft']=='Preserve this failed-start draft.' and f['cwd']=='~' and f['status'], f
assert r['model'] in f['model'] and r['reasoningEffort'] in f['effort'] and f['record']==r, f
m=read('mobile'); assert not m['sidebarOpen'] and not m['overflow'] and m['focused']=='msg', m
(p/'report.json').write_text(json.dumps({'commit':sys.argv[2], 'homeCwd':s['cwd'],'launch':r,'draftHandoff':True,'sendAcknowledgedByBroker':True,'assistantReplyRecorded':assistant_reply,'providerErrors':provider_errors,'sendEchoVisible':True,'ordinaryDialogSendAcknowledged':True,'failurePreservedDraft':True,'mobile':m},indent=2))
print('PASS: actual dialog → quick launch → draft → broker ACK + Pi PTY receipt; server-error fallback; narrow drawer/focus')
if not assistant_reply: print('BOUNDARY: no assistant reply observed; provider diagnostics are in report.json and broker-evidence.json')
PY
printf 'Artifacts: %s\n' "$artifacts"
