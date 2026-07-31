#!/usr/bin/env bash
# Drive the UNATTED stale-interrupted-idle discriminator on port 19268.
#
# Boundary under test: the real server unattended sweep must NOT inject while
# /api/sessions (sidebar) reports busy, even when the raw broker socket keeps
# returning {busy:false, queue_len:0, interrupted_idle:true}.
#
# The discriminator log isolates the readiness gate as the sole blocker:
#   baseline: user_message + non-final assistant fragment (interrupted turn)
#   appended: task_complete(old ts, last_agent_message="done")
#             agent_reasoning(later ts)
# -> _compute_idle_from_log = False (busy, latest agent_reasoning)
# -> _last_chat_role_ts_from_tail(final_assistant_only=True) = ("assistant", task_complete_ts)
#    so the unattended TAIL gate passes; ONLY the readiness gate can stop the sweep.
#
# PASS : zero cmd:send/cmd:keys in broker call log; unattended GET still
#        remaining_injections=1/enabled=true; listing still busy.
# DEFECT: any cmd:send/cmd:keys, OR remaining_injections decrements to 0 /
#         enabled flips to false due to a successful injection while busy.
set -u
PORT=19268
SID=cert-unattended-stale
CONT=codoxear-sandbox-19268
ART=/tmp/codoxear-docker-sandbox-19268/artifacts
REPO_ART=/home/yiwen/codex-web-product-recovery/.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/unattended-stale-interrupted-idle-19268
JAR=$ART/cookies.txt
LOG=/home/tester/cert-logs/cert-unattended-stale.jsonl

mkdir -p "$ART"

curl -sS -c $JAR -H 'Content-Type: application/json' -d '{"password":"test-password"}' \
  "http://127.0.0.1:$PORT/api/login" >/dev/null

broker_state() {
  docker exec $CONT python3 -c "
import socket,json
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect('/home/tester/.local/share/codoxear/socks/${SID}.sock')
s.sendall((json.dumps({'cmd':'state'})+chr(10)).encode())
buf=b''
while True:
    c=s.recv(4096)
    if not c: break
    buf+=c
    if b'\n' in buf: break
print(buf.decode().strip())
"
}

call_log_summary() {
  docker exec $CONT python3 -c "
import json
calls=[]
try:
    for line in open('/tmp/unattended_broker_calls.jsonl'):
        line=line.strip()
        if line: calls.append(json.loads(line))
except FileNotFoundError: pass
sends=[c for c in calls if c['cmd']=='send']
keys=[c for c in calls if c['cmd']=='keys']
states=[c for c in calls if c['cmd']=='state']
others=[c for c in calls if c['cmd'] not in ('send','keys','state','tail')]
print(json.dumps({'total':len(calls),'states':len(states),'sends':len(sends),
                  'keys':len(keys),'others':len(others),'send_details':sends,'key_details':keys}))
"
}

probe_busy() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);s=[x for x in d['sessions'] if x['session_id']=='${SID}'];print(json.dumps({'busy':s[0]['busy'],'missing':False,'unattended_enabled':s[0].get('unattended_enabled'),'unattended_remaining':s[0].get('unattended_remaining_injections')} if s else {'missing':True}))"
}

full_row() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);s=[x for x in d['sessions'] if x['session_id']=='${SID}'];print(json.dumps(s[0] if s else {'missing':True},indent=2))"
}

unattended_get() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions/${SID}/unattended"
}

logsize() { docker exec $CONT wc -c "$LOG" | awk '{print $1}'; }

append_row() {
  docker exec -i $CONT sh -c "cat >> '$LOG'" <<<"$1"
}

echo "============================================================"
echo "PHASE A: interrupted non-final log + broker interrupted_idle=true"
echo "  EXPECTED: listing busy=false (override valid), raw interrupted_idle=true"
echo "============================================================"
# wait for discovery
for i in $(seq 1 30); do
  B=$(probe_busy); echo "$B" | grep -q missing || break; sleep 1
done
BS_A=$(broker_state)
echo "broker_state: $BS_A"
echo "log_size: $(logsize)"
PB_A=$(probe_busy); echo "api_busy: $PB_A"
echo "$BS_A" > $REPO_ART/phaseA-broker-state.json
full_row > $REPO_ART/phaseA-sessions.json

echo ""
echo "============================================================"
echo "PHASE B: append task_complete(old ts) + agent_reasoning(later ts)"
echo "  Broker STILL raw interrupted_idle=true. Listing should go busy=true"
echo "  (suppression clears stored override; _compute_idle_from_log=False)."
echo "  Tail gate still passes (assistant, old ts) -> readiness is sole gate."
echo "============================================================"
append_row '{"type": "event_msg", "ts": 12.0, "payload": {"type": "task_complete", "last_agent_message": "done"}}'
append_row '{"type": "event_msg", "ts": 13.0, "payload": {"type": "agent_reasoning", "message": "reasoning resumed after interrupt"}}'
echo "log_size after append: $(logsize)"
BS_B=$(broker_state)
echo "broker_state (raw, still stale): $BS_B"
# poll until busy true
BUSY_HIT=""
for i in $(seq 1 20); do
  P=$(probe_busy); echo "poll $i: $P"
  echo "$P" | grep -q '"busy": true' && { BUSY_HIT="$P"; break; }
  sleep 1
done
PB_B="$BUSY_HIT"
echo "PHASE B divergence confirmed: listing busy while raw broker interrupted_idle=true"
echo "$BS_B" > $REPO_ART/phaseB-broker-state.json
printf '{"busy_polls_last":"%s","broker_state":%s,"log_size":"%s"}\n' "$PB_B" "$BS_B" "$(logsize)" > $REPO_ART/phaseB-polls.json
full_row > $REPO_ART/phaseB-sessions-final.json

echo ""
echo "============================================================"
echo "PHASE C: enable unattended via REAL API"
echo "  POST /api/sessions/<sid>/unattended"
echo "  {enabled:true, request:'unattended stale busy probe',"
echo "   cooldown_minutes:1, remaining_injections:1}"
echo "============================================================"
UNATT_BEFORE=$(unattended_get); echo "unattended GET before: $UNATT_BEFORE"
echo "$UNATT_BEFORE" > $REPO_ART/phaseC-unattended-before.json
HTTP_UNATT=$(curl -sS -o $REPO_ART/phaseC-unattended-enable-body.txt -w '%{http_code}' \
  -b $JAR -H 'Content-Type: application/json' \
  -d '{"enabled":true,"request":"unattended stale busy probe","cooldown_minutes":1,"remaining_injections":1}' \
  "http://127.0.0.1:$PORT/api/sessions/${SID}/unattended")
echo "unattended POST HTTP status: $HTTP_UNATT"
echo "unattended POST body:"; cat $REPO_ART/phaseC-unattended-enable-body.txt; echo
printf '{"http_code":"%s","response_body":' "$HTTP_UNATT" > $REPO_ART/phaseC-unattended-enable.json
cat $REPO_ART/phaseC-unattended-enable-body.txt >> $REPO_ART/phaseC-unattended-enable.json
echo "}" >> $REPO_ART/phaseC-unattended-enable.json
UNATT_AFTER=$(unattended_get); echo "unattended GET after enable: $UNATT_AFTER"
echo "$UNATT_AFTER" > $REPO_ART/phaseC-unattended-after.json

echo ""
echo "============================================================"
echo "PHASE D: let the REAL unattended sweep run (2.5s period; wait 12s ~5 sweeps)"
echo "  PASS : zero cmd:send/cmd:keys; unattended still remaining=1/enabled=true; busy"
echo "  DEFECT: any cmd:send/cmd:keys, OR remaining->0/enabled->false while busy"
echo "============================================================"
CALLS_BEFORE=$(call_log_summary); echo "call_log before sweep wait: $CALLS_BEFORE"
echo "$CALLS_BEFORE" > $REPO_ART/phaseD-calllog-before.json
echo "waiting 12s for unattended sweep to run ~5 times..."
sleep 12
CALLS_AFTER=$(call_log_summary); echo "call_log after sweep wait: $CALLS_AFTER"
echo "$CALLS_AFTER" > $REPO_ART/phaseD-calllog-after.json
UNATT_FINAL=$(unattended_get); echo "unattended GET final: $UNATT_FINAL"
echo "$UNATT_FINAL" > $REPO_ART/phaseD-unattended-final.json
PB_D=$(probe_busy); echo "listing busy final: $PB_D"
echo "$PB_D" > $REPO_ART/phaseD-sessions-busy.txt
full_row > $REPO_ART/phaseD-sessions-final.json
echo "raw broker state final: $(broker_state)"

echo ""
echo "============================================================"
echo "PHASE D: VERDICT"
echo "============================================================"
python3 - "$REPO_ART" "$CALLS_AFTER" "$UNATT_FINAL" "$PB_D" <<'PY'
import json, sys, re
repo_art = sys.argv[1]
calls = json.loads(sys.argv[2])
unatt = json.loads(sys.argv[3])
busy_txt = sys.argv[4]

sends = calls.get("sends", 0)
keys = calls.get("keys", 0)
enabled = unatt.get("enabled")
remaining = unatt.get("remaining_injections")
busy_match = re.search(r'"busy": (true|false)', busy_txt)
busy = (busy_match.group(1) == "true") if busy_match else None

injected = sends > 0 or keys > 0
decremented = (remaining == 0)
disabled = (enabled is False)

defect = bool(injected or decremented or disabled)
verdict = "DEFECT" if defect else "PASS"
# Additional diagnostic: if not busy, the precondition itself failed.
precondition_ok = (busy is True)
print(f"sends={sends} keys={keys} enabled={enabled} remaining={remaining} busy={busy}")
print(f"injected={injected} decremented={decremented} disabled={disabled} precondition_busy={precondition_ok}")
print(f"VERDICT={verdict}")
out = {
  "verdict": verdict,
  "sends": sends,
  "keys": keys,
  "total_calls": calls.get("total"),
  "state_calls": calls.get("states"),
  "unattended_enabled": enabled,
  "unattended_remaining_injections": remaining,
  "listing_busy": busy,
  "precondition_busy_held": precondition_ok,
  "injected_while_busy": injected,
  "remaining_decremented": decremented,
  "disabled_by_success": disabled,
}
with open(f"{repo_art}/verdict.json", "w") as f:
    json.dump(out, f, indent=2)
PY

echo ""
echo "============================================================"
echo "FINAL: full broker call log dump"
echo "============================================================"
docker exec $CONT cat /tmp/unattended_broker_calls.jsonl 2>/dev/null > $REPO_ART/broker-call-log.jsonl || true
echo "(call log lines: $(wc -l < $REPO_ART/broker-call-log.jsonl))"
echo "DONE"
