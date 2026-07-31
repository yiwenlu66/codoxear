#!/usr/bin/env bash
# Drive phases 1-3 against the real Codoxear server on port 19250.
# Appends use `docker exec -i` so stdin reaches the inner shell.
set -u
PORT=19250
SID=cert-stale-interrupt
ART=/tmp/codoxear-docker-sandbox-19250/artifacts
REPO_ART=/home/yiwen/codex-web-product-recovery/.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/log-only-stale-interrupted-idle-19250
JAR=$ART/cookies.txt
LOG=/home/tester/cert-logs/cert-stale-interrupt.jsonl
CONT=codoxear-sandbox-19250

curl -sS -c $JAR -H 'Content-Type: application/json' -d '{"password":"test-password"}' "http://127.0.0.1:$PORT/api/login" >/dev/null

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

probe_busy() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);s=[x for x in d['sessions'] if x['session_id']=='${SID}'];print(json.dumps({'busy':s[0]['busy'],'missing':False} if s else {'missing':True}))"
}

full_row() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);s=[x for x in d['sessions'] if x['session_id']=='${SID}'];print(json.dumps(s[0] if s else {'missing':True},indent=2))"
}

logsize() { docker exec $CONT wc -c "$LOG" | awk '{print $1}'; }

append_row() {
  # $1 = JSON object string (one line)
  docker exec -i $CONT sh -c "cat >> '$LOG'" <<<"$1"
}

echo "============================================================"
echo "PHASE 1: interrupted non-final log + broker interrupted_idle=true"
echo "============================================================"
echo "broker_state: $(broker_state)"
echo "log_size: $(logsize)"
echo "api_busy: $(probe_busy)"
full_row > $REPO_ART/phase1-sessions.json
echo "(full row -> phase1-sessions.json)"

echo ""
echo "============================================================"
echo "PHASE 2: append post-interrupt user_message; broker STILL interrupted_idle=true"
echo "============================================================"
append_row '{"type": "event_msg", "ts": 20.0, "payload": {"type": "user_message", "message": "resumed turn after interrupt"}}'
echo "log_size after append: $(logsize)"
echo "broker_state: $(broker_state)"
P1=$(probe_busy); echo "poll1: $P1"; sleep 1
P2=$(probe_busy); echo "poll2: $P2"; sleep 1
P3=$(probe_busy); echo "poll3: $P3"; sleep 1
P4=$(probe_busy); echo "poll4: $P4"; sleep 1
P5=$(probe_busy); echo "poll5: $P5"
printf '{"polls":[%s,%s,%s,%s,%s],"broker_state":"%s","log_size":"%s"}\n' "$P1" "$P2" "$P3" "$P4" "$P5" "$(broker_state)" "$(logsize)" > $REPO_ART/phase2-polls.json
full_row > $REPO_ART/phase2-sessions-final.json

echo ""
echo "============================================================"
echo "PHASE 3a (optional): flip broker to interrupted_idle=false"
echo "============================================================"
docker exec $CONT sh -c 'echo false > /tmp/stale_broker_ctrl'
sleep 1
echo "broker_state: $(broker_state)"
A1=$(probe_busy); echo "api_busy (override cleared, log non-idle -> expect true): $A1"; sleep 1
A2=$(probe_busy); echo "api_busy poll2: $A2"
printf '{"polls":[%s,%s],"broker_state":"%s"}\n' "$A1" "$A2" "$(broker_state)" > $REPO_ART/phase3a-polls.json

echo ""
echo "============================================================"
echo "PHASE 3b (optional): fresh interrupt at later offset, re-arm, then post-arm resume"
echo "============================================================"
append_row '{"type": "event_msg", "ts": 30.0, "payload": {"type": "user_message", "message": "second turn that will be interrupted"}}'
append_row '{"type": "response_item", "ts": 31.0, "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "start"}]}}'
echo "log_size after fresh interrupt turn: $(logsize)"
docker exec $CONT sh -c 'echo true > /tmp/stale_broker_ctrl'
sleep 1
echo "broker_state (re-armed true): $(broker_state)"
B0=$(probe_busy); echo "api_busy right after re-arm: $B0"
append_row '{"type": "event_msg", "ts": 40.0, "payload": {"type": "user_message", "message": "resumed again after second interrupt"}}'
echo "log_size after post-arm resume: $(logsize)"
sleep 1
B1=$(probe_busy); echo "post-arm poll1: $B1"; sleep 1
B2=$(probe_busy); echo "post-arm poll2: $B2"
printf '{"rearm_busy":"%s","post_arm_polls":[%s,%s],"broker_state":"%s","log_size":"%s"}\n' "$B0" "$B1" "$B2" "$(broker_state)" "$(logsize)" > $REPO_ART/phase3b-polls.json
full_row > $REPO_ART/phase3-sessions-final.json
echo "DONE"
