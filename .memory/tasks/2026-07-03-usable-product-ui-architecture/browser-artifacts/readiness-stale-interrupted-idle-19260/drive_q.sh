#!/usr/bin/env bash
# Clean queue discriminator: fresh session (SID cert-stale-q), queue probed
# BEFORE any direct send, so no leftover send-boundary can mask the divergence.
set -u
PORT=19260
SID=cert-stale-q
CONT=codoxear-sandbox-19260
ART=/tmp/codoxear-docker-sandbox-19260/artifacts
REPO_ART=/home/yiwen/codex-web-product-recovery/.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/readiness-stale-interrupted-idle-19260
JAR=$ART/cookies.txt
LOG=/home/tester/cert-logs/cert-stale-q.jsonl

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

calllog_summary() {
  docker exec $CONT python3 -c "
import json
calls=[]
try:
    for line in open('/tmp/stale_broker_calls_q.jsonl'):
        line=line.strip()
        if line: calls.append(json.loads(line))
except FileNotFoundError: pass
sends=[c for c in calls if c['cmd']=='send']
print(json.dumps({'total':len(calls),'sends':len(sends),'send_details':sends}))
"
}

probe_busy() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);s=[x for x in d['sessions'] if x['session_id']=='${SID}'];print(json.dumps({'busy':s[0]['busy'],'missing':False} if s else {'missing':True}))"
}

append_row() {
  docker exec -i $CONT sh -c "cat >> '$LOG'" <<<"$1"
}

logsize() { docker exec $CONT wc -c "$LOG" | awk '{print $1}'; }

echo "waiting for discovery of $SID ..."
for i in $(seq 1 20); do
  B=$(probe_busy)
  echo "$B" | grep -q missing && { sleep 1; continue; }
  break
done
echo "discovered: $B"

echo ""
echo "PHASE Q2: append post-interrupt user_message; broker STILL interrupted_idle=true"
append_row '{"type": "event_msg", "ts": 20.0, "payload": {"type": "user_message", "message": "resumed turn after interrupt"}}'
echo "log_size: $(logsize)"
echo "broker_state: $(broker_state)"
sleep 2
B1=$(probe_busy); echo "poll1 (expect busy=true): $B1"; sleep 1
B2=$(probe_busy); echo "poll2: $B2"
printf '{"polls":[%s,%s],"broker_state":%s,"log_size":"%s"}\n' "$B1" "$B2" "$(broker_state)" "$(logsize)" > $REPO_ART/phaseQ2-polls.json

echo ""
echo "PHASE Q5: QUEUE DISCRIMINATOR (fresh session, queue before any send)"
echo "  EXPECTED PASS: item queued, broker receives NO cmd:send"
echo "  DEFECT: broker records cmd:send (queue promotion) while sidebar busy"
BEFORE=$(calllog_summary)
echo "calllog before enqueue: $BEFORE"
HTTP=$(curl -sS -o $REPO_ART/phaseQ5-enqueue-body.txt -w '%{http_code}' \
  -b $JAR -H 'Content-Type: application/json' \
  -d '{"text":"queue probe on fresh busy session"}' \
  "http://127.0.0.1:$PORT/api/sessions/${SID}/enqueue")
echo "enqueue HTTP: $HTTP"
echo "body: $(cat $REPO_ART/phaseQ5-enqueue-body.txt)"
sleep 1
AFTER=$(calllog_summary)
echo "calllog after enqueue: $AFTER"
printf '{"http_code":"%s","before":%s,"after":%s,"body":' "$HTTP" "$BEFORE" "$AFTER" > $REPO_ART/phaseQ5-enqueue-result.json
cat $REPO_ART/phaseQ5-enqueue-body.txt >> $REPO_ART/phaseQ5-enqueue-result.json
echo "}" >> $REPO_ART/phaseQ5-enqueue-result.json
curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions/${SID}/queue" > $REPO_ART/phaseQ5-queue-get.json
echo "queue GET: $(cat $REPO_ART/phaseQ5-queue-get.json)"
echo "FINAL busy: $(probe_busy)"
echo "FINAL broker: $(broker_state)"
echo "DONE"
