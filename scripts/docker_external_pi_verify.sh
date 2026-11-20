#!/usr/bin/env bash
# Real Pi + global extension + local streaming provider; all processes/data in Docker.
# Working-tree build by default, or pass a reviewed commit to verify its archive.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
port="${CODOXEAR_EXTERNAL_VERIFY_PORT:-19721}"
[[ "$port" != 8743 ]] || { echo 'refusing live port' >&2; exit 2; }
root="$(mktemp -d /tmp/codoxear-external-pi.XXXXXX)"
artifacts="${CODOXEAR_EXTERNAL_VERIFY_ARTIFACTS:-$(mktemp -d /tmp/codoxear-external-pi-results.XXXXXX)}"
container="codoxear-external-pi-$port-$$"
image="codoxear-external-pi:verify"
mkdir -p "$root/context" "$artifacts"
cleanup() {
  if command -v agent-browser >/dev/null; then
    AGENT_BROWSER_SESSION="external-pi-$$" agent-browser close >/dev/null 2>&1 || true
  fi
  sudo docker logs "$container" > "$artifacts/container.log" 2>&1 || true
  sudo docker cp "$container:/home/tester/proof" "$artifacts/proof" >/dev/null 2>&1 || true
  sudo docker rm -f "$container" >/dev/null 2>&1 || true
  echo "artifacts=$artifacts"
}
trap cleanup EXIT
if [[ -n "${1:-}" ]]; then
  git -C "$repo" archive "$1" | tar -x -C "$root/context"
else
  git -C "$repo" ls-files -z --cached --others --exclude-standard | python3 -c 'import sys; sys.stdout.buffer.write(b"\0".join(p for p in sys.stdin.buffer.read().split(b"\0") if p and not p.startswith((b".memory/",b"review/")))+b"\0")' > "$root/files"
  tar --null --no-recursion -C "$repo" -T "$root/files" -cf - | tar -xf - -C "$root/context"
fi
# Keep runtime tools self-contained, including console entry points and test Node.
cat >> "$root/context/docker/verify.Dockerfile" <<'DOCKER'
USER root
RUN apt-get update && apt-get install -y tmux && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/local/bin/node /usr/bin/node \
    && pip install --no-deps -e /workspace \
    && npm install --global esbuild \
    && esbuild codoxear/static/app.js --bundle --minify --format=esm --outfile=codoxear/static/dist/app.bundle.js
USER tester
DOCKER
sudo docker build --build-arg PI_VERSION=1.5.2 -f "$root/context/docker/verify.Dockerfile" -t "$image" "$root/context" > "$artifacts/build.log" 2>&1
sudo docker run -d --name "$container" -p "127.0.0.1:$port:$port" \
  -e CODEX_WEB_PASSWORD=external-test-password -e CODEX_WEB_HOST=0.0.0.0 -e CODEX_WEB_PORT="$port" \
  -e TERM=xterm-256color -e PI_OFFLINE=1 \
  "$image" python3 scripts/docker_external_pi_verify.py --port "$port" > "$artifacts/container-id.txt"
for _ in $(seq 1 240); do
  if sudo docker exec "$container" test -f /home/tester/proof/ready; then break; fi
  if [[ "$(sudo docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
    sudo docker logs "$container"; exit 1
  fi
  sleep 1
done
sudo docker exec "$container" test -f /home/tester/proof/ready
# Optional real browser proof: actual selection, composer send, live reply,
# hidden inapplicable controls, External last, narrow viewport geometry.
if command -v agent-browser >/dev/null; then
  export AGENT_BROWSER_SESSION="external-pi-$$"
  agent-browser open "http://127.0.0.1:$port/" >/dev/null
  agent-browser eval 'document.querySelector("input[type=password]").value="external-test-password"; document.querySelector("input[type=password]").dispatchEvent(new Event("input",{bubbles:true})); document.querySelector("button[type=submit]").click()' >/dev/null
  sleep 2
  agent-browser eval 'document.querySelector("[data-session-id^=external-pi-]").click()' > "$artifacts/browser-select.json"
  sleep 2
  agent-browser eval 'document.querySelector("#diagBtn").click()' >/dev/null
  sleep 1
  agent-browser eval 'if(document.querySelector("#diagStatus").textContent || !document.querySelector("#diagContent").textContent.includes("External (Pi-owned)")) throw Error("External details failed"); document.querySelector("#diagCloseBtn").click()' > "$artifacts/browser-details.json"
  agent-browser eval 'const t=document.querySelector("#msg"); t.value="/model"; t.dispatchEvent(new Event("input",{bubbles:true})); if(document.querySelector("#modelPicker").getBoundingClientRect().width) throw Error("External model picker exposed"); t.value="BROWSER real user message"; t.dispatchEvent(new Event("input",{bubbles:true})); document.querySelector("#sendBtn").click()' > "$artifacts/browser-send.json"
  sleep 3
  agent-browser eval 'const proof={externalLast:[...document.querySelectorAll(".sessionGroupLabel")].at(-1)?.textContent,reply:document.querySelector("#chat").textContent.includes("LOCAL RESPONSE") && document.querySelector("#chat").textContent.includes("BROWSER"),controls:["attachBtn","unattendedBtn","queueBtn"].map(id=>[id,document.getElementById(id)?.getBoundingClientRect().width]),overflow:document.documentElement.scrollWidth>innerWidth}; if(proof.externalLast!=="External"||!proof.reply||proof.controls.some(([,w])=>w>0)||proof.overflow) throw Error(JSON.stringify(proof)); JSON.stringify(proof)' > "$artifacts/browser-state.json"
  agent-browser eval 'window.__externalSelectedId=document.querySelector(".session.active").dataset.sessionId' > "$artifacts/browser-before-new.json"
  sudo docker exec "$container" tmux send-keys -t external C-u /new Enter
  sleep 5
  agent-browser eval 'const current=document.querySelector(".session.active"); if(!current||current.dataset.sessionId!==window.__externalSelectedId||document.querySelector("#msg").disabled||document.querySelector("#chat").textContent.includes("BROWSER real user message")) throw Error("Selected external row or transcript failed to follow /new"); const t=document.querySelector("#msg"); t.value="BROWSER after new"; t.dispatchEvent(new Event("input",{bubbles:true})); document.querySelector("#sendBtn").click()' > "$artifacts/browser-after-new.json"
  sleep 3
  agent-browser eval 'if(!document.querySelector("#chat").textContent.includes("BROWSER after new")) throw Error("Replacement transcript did not receive user message")' >/dev/null
  agent-browser eval 'const t=document.querySelector("#msg"); t.value="SLOW browser interrupt"; t.dispatchEvent(new Event("input",{bubbles:true})); document.querySelector("#sendBtn").click()' >/dev/null
  sleep 2
  agent-browser eval 'const b=document.querySelector("#interruptBtn"); if(!b.getBoundingClientRect().width||b.disabled) throw Error("Interrupt unavailable"); b.click()' > "$artifacts/browser-interrupt.json"
  sleep 2
  agent-browser screenshot "$artifacts/external-desktop.png" >/dev/null
  agent-browser set viewport 393 844 >/dev/null
  agent-browser screenshot "$artifacts/external-mobile.png" >/dev/null
  agent-browser eval 'if(document.documentElement.scrollWidth>innerWidth) throw Error("Mobile overflow"); JSON.stringify({overflow:false})' > "$artifacts/browser-mobile.json"
  agent-browser close >/dev/null
fi
sudo docker exec "$container" touch /home/tester/proof/finish
for _ in $(seq 1 90); do
  [[ "$(sudo docker inspect -f '{{.State.Running}}' "$container")" == true ]] || break
  sleep 1
done
sudo docker logs "$container"
[[ "$(sudo docker inspect -f '{{.State.ExitCode}}' "$container")" == 0 ]]
sudo docker cp "$container:/home/tester/proof/report.json" "$artifacts/report.json"
echo 'PASS: multi-session Pi SDK, lazy registration/disposal, terminal/broker controls and lifecycle'
