#!/usr/bin/env bash
# Drive the send/queue/readiness divergence discriminator on port 19264 (FIXED HEAD 206fb6c).
# Phases 1-2 establish the listing/sidebar busy divergence (broker keeps
# interrupted_idle:true while a post-interrupt user row lands on the log).
# Phase 4 probes the DIRECT send route; phase 5 probes the QUEUE promotion route
# (same session, after the send boundary).
set -u
PORT=19264
SID=cert-stale-interrupt
CONT=codoxear-sandbox-19264
ART=/tmp/codoxear-docker-sandbox-19264/artifacts
REPO_ART=/home/yiwen/codex-web-product-recovery/.memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/readiness-stale-interrupted-idle-fixed-19264
JAR=$ART/cookies.txt
LOG=/home/tester/cert-logs/cert-stale-interrupt.jsonl

mkdir -p "$ART"

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

call_log_summary() {
  docker exec $CONT python3 -c "
import json
calls=[]
try:
    for line in open('/tmp/stale_broker_calls.jsonl'):
        line=line.strip()
        if line: calls.append(json.loads(line))
except FileNotFoundError: pass
sends=[c for c in calls if c['cmd']=='send']
keys=[c for c in calls if c['cmd']=='keys']
print(json.dumps({'total':len(calls),'sends':len(sends),'keys':len(keys),'send_details':sends}))
"
}

call_log_dump() {
  docker exec $CONT cat /tmp/stale_broker_calls.jsonl 2>/dev/null || true
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
  docker exec -i $CONT sh -c "cat >> '$LOG'" <<<"$1"
}

echo "============================================================"
echo "PHASE 1: interrupted non-final log + broker interrupted_idle=true"
echo "============================================================"
# wait for discovery
for i in $(seq 1 20); do
  B=$(probe_busy); echo "$B" | grep -q missing || break; sleep 1
done
echo "broker_state: $(broker_state)"
echo "log_size: $(logsize)"
echo "api_busy: $(probe_busy)"
full_row > $REPO_ART/phase1-sessions.json

echo ""
echo "============================================================"
echo "PHASE 2: append post-interrupt user_message; broker STILL true"
echo "============================================================"
append_row '{"type": "event_msg", "ts": 20.0, "payload": {"type": "user_message", "message": "resumed turn after interrupt"}}'
echo "log_size after append: $(logsize)"
echo "broker_state: $(broker_state)"
P1=$(probe_busy); echo "poll1: $P1"; sleep 1
P2=$(probe_busy); echo "poll2: $P2"; sleep 1
P3=$(probe_busy); echo "poll3: $P3"
printf '{"polls":[%s,%s,%s],"broker_state":%s,"log_size":"%s"}\n' "$P1" "$P2" "$P3" "$(broker_state)" "$(logsize)" > $REPO_ART/phase2-polls.json
full_row > $REPO_ART/phase2-sessions-final.json

echo ""
echo "PHASE 2 assertion: sidebar busy=true while broker interrupted_idle=true"
BS=$(broker_state)
echo "broker_state now: $BS"

echo ""
echo "============================================================"
echo "PHASE 4: DIRECT SEND DISCRIMINATOR"
echo "  POST /api/sessions/<sid>/send"
echo "  EXPECTED (PASS): HTTP 409 not-ready, broker receives NO send attempt"
echo "  DEFECT: HTTP 200 + broker records a cmd:send attempt"
echo "============================================================"
CALLS_BEFORE=$(call_log_summary)
echo "call_log before send: $CALLS_BEFORE"
HTTP_CODE=$(curl -sS -o $REPO_ART/phase4-send-body.txt -w '%{http_code}' \
  -b $JAR -H 'Content-Type: application/json' \
  -d '{"text":"probe direct send while sidebar busy FIXED"}' \
  "http://127.0.0.1:$PORT/api/sessions/${SID}/send")
echo "send HTTP status: $HTTP_CODE"
echo "send response body:"
cat $REPO_ART/phase4-send-body.txt
echo ""
sleep 1
CALLS_AFTER=$(call_log_summary)
echo "call_log after send: $CALLS_AFTER"
printf '{"http_code":"%s","calls_before":%s,"calls_after":%s,"response_body":' "$HTTP_CODE" "$CALLS_BEFORE" "$CALLS_AFTER" > $REPO_ART/phase4-send-result.json
cat $REPO_ART/phase4-send-body.txt >> $REPO_ART/phase4-send-result.json
echo "}" >> $REPO_ART/phase4-send-result.json

echo ""
echo "============================================================"
echo "PHASE 5: QUEUE (same session, after send boundary)"
echo "  EXPECTED: item queued (sidebar busy), zero cmd:send"
echo "============================================================"
CALLS_BEFORE_Q=$(call_log_summary)
echo "call_log before enqueue: $CALLS_BEFORE_Q"
HTTP_CODE_Q=$(curl -sS -o $REPO_ART/phase5-enqueue-body.txt -w '%{http_code}' \
  -b $JAR -H 'Content-Type: application/json' \
  -d '{"text":"probe queue promotion while sidebar busy FIXED"}' \
  "http://127.0.0.1:$PORT/api/sessions/${SID}/enqueue")
echo "enqueue HTTP status: $HTTP_CODE_Q"
echo "enqueue response body:"
cat $REPO_ART/phase5-enqueue-body.txt
echo ""
sleep 1
CALLS_AFTER_Q=$(call_log_summary)
echo "call_log after enqueue: $CALLS_AFTER_Q"
printf '{"http_code":"%s","calls_before":%s,"calls_after":%s,"response_body":' "$HTTP_CODE_Q" "$CALLS_BEFORE_Q" "$CALLS_AFTER_Q" > $REPO_ART/phase5-enqueue-result.json
cat $REPO_ART/phase5-enqueue-body.txt >> $REPO_ART/phase5-enqueue-result.json
echo "}" >> $REPO_ART/phase5-enqueue-result.json

curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions/${SID}/queue" \
  > $REPO_ART/phase5-queue-get.json
echo "queue GET:"
cat $REPO_ART/phase5-queue-get.json
echo ""

echo ""
echo "============================================================"
echo "FINAL: full call log dump (broker 1)"
echo "============================================================"
call_log_dump > $REPO_ART/broker1-call-log.jsonl
cat $REPO_ART/broker1-call-log.jsonl
echo ""
echo "FINAL api_busy: $(probe_busy)"
echo "FINAL broker_state: $(broker_state)"
echo "DONE"
