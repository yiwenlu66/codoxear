#!/usr/bin/env bash
# Claim 5: drive cert-interrupt through interrupt-idle -> resumed(busy) -> complete(idle)
set -u
ART=/tmp/codoxear-cert-19200/cert-artifacts
JAR=$ART/cookies.txt
PORT=19200
SID=cert-interrupt
OUT=$ART/claim5-api-timeline.json

probe() {
  curl -sS -b $JAR "http://127.0.0.1:$PORT/api/sessions" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);s=[x for x in d['sessions'] if x['session_id']=='$SID'];print(json.dumps({'busy':s[0]['busy']} if s else {'missing':True}))" 2>/dev/null
}

{
  echo "{"
  echo "\"phase0_post_interrupt_idle\": $(probe),"

  # Trigger resumed activity via /send (broker flips busy=True, interrupted_idle=False).
  SEND=$(curl -sS -b $JAR -X POST -H 'Content-Type: application/json' -d '{"text":"resume the work","allow_pending_attachment":false}' "http://127.0.0.1:$PORT/api/sessions/$SID/send")
  echo "\"send_response\": $SEND,"

  sleep 3
  echo "\"phase1_resumed_running\": $(probe),"

  # The stateful broker auto-completes the turn ~4s after send; wait for it.
  sleep 5
  echo "\"phase2_completed_idle\": $(probe)"
  echo "}"
} | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin),indent=2))" > $OUT
cat $OUT
