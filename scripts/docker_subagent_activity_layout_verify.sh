#!/usr/bin/env bash
# Docker-isolated real API→UI geometry regression for subagent activity headers.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${CODOXEAR_SUBAGENT_LAYOUT_PORT:-19671}"
password="${CODOXEAR_SUBAGENT_LAYOUT_PASSWORD:-subagent-layout-password}"
image="codoxear-subagent-layout:working-tree"
container="codoxear-subagent-layout-${port}"
root="$(mktemp -d /tmp/codoxear-subagent-layout-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-subagent-layout-results.XXXXXX)"
home_dir="$root/home"
browser_session="codoxear-subagent-layout-$$"
readonly repo_root port password image container root artifacts home_dir browser_session
readonly -a docker=(sudo docker)

if [[ "$port" == "8743" ]]; then
  echo "refusing live Codoxear port 8743" >&2
  exit 2
fi
if [[ -n "${CODEXEAR_APP_DIR:-}${CODEX_WEB_APP_DIR:-}" ]]; then
  echo "refusing host app-dir override" >&2
  exit 2
fi
if ! command -v agent-browser >/dev/null; then
  echo "agent-browser is required" >&2
  exit 2
fi

browser() {
  AGENT_BROWSER_SESSION="$browser_session" agent-browser "$@"
}

cleanup() {
  browser close >/dev/null 2>&1 || true
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  "${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true
  echo "artifacts=$artifacts" >&2
  exit 1
}

mkdir -p "$home_dir/.local/share"
sudo chown -R 1000:1000 "$home_dir"
"${docker[@]}" build --file "$repo_root/docker/sandbox.Dockerfile" --tag "$image" "$repo_root" \
  > "$artifacts/docker-build.log" 2>&1 || fail "Docker image build failed"
"${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker[@]}" run --detach \
  --name "$container" \
  --publish "127.0.0.1:${port}:${port}" \
  --env HOME=/home/tester \
  --env CODEX_WEB_PASSWORD="$password" \
  --env CODEX_WEB_HOST=0.0.0.0 \
  --env CODEX_WEB_PORT="$port" \
  --env CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS=0.1 \
  --env CODEX_WEB_SUBAGENT_RUNS_ROOT=/home/tester/subagent-runs \
  --env PYTHONPATH=/workspace \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$repo_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$home_dir,dst=/home/tester" \
  --workdir /workspace \
  "$image" \
  bash -c 'python3 tests/fixtures/subagent_activity_broker.py >"$HOME/fixture.log" 2>&1 & exec python3 -m codoxear.server' \
  > "$artifacts/container-id.txt" || fail "container start failed"

for _ in $(seq 1 120); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/api/me" 2>/dev/null || true)"
  [[ "$code" == "401" ]] && break
  sleep 0.25
done
[[ "${code:-}" == "401" ]] || fail "server readiness failed"

cookie="$root/cookies.txt"
curl -fsS -c "$cookie" -H 'Content-Type: application/json' \
  --data "{\"password\":\"${password}\"}" "http://127.0.0.1:${port}/api/login" > "$artifacts/api-login.json"
for _ in $(seq 1 80); do
  curl -fsS -b "$cookie" "http://127.0.0.1:${port}/api/sessions" > "$artifacts/api-sessions.json" || true
  if python3 - "$artifacts/api-sessions.json" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
rows = payload.get("sessions", []) if isinstance(payload, dict) else []
raise SystemExit(0 if any(row.get("session_id") == "subagent-layout" and row.get("subagents_running") == 2 for row in rows if isinstance(row, dict)) else 1)
PY
  then
    break
  fi
  sleep 0.25
done
python3 - "$artifacts/api-sessions.json" <<'PY' || fail "API fixture did not expose two children"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert any(row.get("session_id") == "subagent-layout" and len(row.get("subagent_details", [])) == 2 for row in payload["sessions"])
PY

browser set viewport 1280 800 >/dev/null 2>&1 || fail "could not set initial viewport"
browser open "http://127.0.0.1:${port}/" > "$artifacts/browser-open.txt" 2>&1 || fail "browser open failed"
browser errors --clear --json >/dev/null 2>&1 || true
browser console --clear --json >/dev/null 2>&1 || true
browser fill '#pw' "$password" >/dev/null 2>&1 || fail "login fill failed"
browser click '#loginBtn' >/dev/null 2>&1 || fail "login click failed"
browser wait 2500 >/dev/null 2>&1
browser click '#sessions .session' >/dev/null 2>&1 || fail "session selection failed"
browser wait 1500 >/dev/null 2>&1

PROBE='(() => {
  const busy = document.querySelector(".msg.typing");
  const idle = document.querySelector(".subagentActivity");
  const bubble = busy || idle;
  if (!bubble) return null;
  const state = busy ? "busy" : "idle";
  const marker = bubble.querySelector(busy ? ".typingDots" : ".subagentActivitySquares");
  const lastDot = bubble.querySelector(busy ? ".typingDot:last-child" : ".subagentActivitySquare:last-child");
  const label = bubble.querySelector(busy ? ".typingStats" : ".subagentActivityText");
  const header = bubble.querySelector(".subagentActivityHeader");
  const details = bubble.querySelector(".subagentDetails");
  const row = bubble.closest(".msg-row");
  const lines = [...bubble.querySelectorAll(".subagentDetailLine")];
  const rect = (node) => { const r = node.getBoundingClientRect(); return { left:r.left, right:r.right, top:r.top, bottom:r.bottom, width:r.width, height:r.height }; };
  const br = rect(bubble), mr = rect(marker), dr = rect(lastDot), lr = rect(label), rr = rect(row), hr = rect(header);
  const bs = getComputedStyle(bubble), ds = details ? getComputedStyle(details) : null;
  const contentLeft = br.left + parseFloat(bs.borderLeftWidth) + parseFloat(bs.paddingLeft);
  const first = lines[0] ? rect(lines[0]) : null;
  return {
    state, viewport: { width: innerWidth, height: innerHeight },
    bubble: { ...br, clientWidth:bubble.clientWidth, scrollWidth:bubble.scrollWidth, gridTemplateColumns:bs.gridTemplateColumns },
    row: rr,
    availableWidth: rr.width,
    padding: { top: parseFloat(bs.paddingTop), left: parseFloat(bs.paddingLeft) },
    contentLeft,
    markerOffsetFromContent: mr.left - contentLeft,
    firstLineOffsetFromContent: first ? first.left - contentLeft : null,
    headerToDetailsGap: details ? rect(details).top - hr.bottom : null,
    intendedHeaderToDetailsGap: ds ? parseFloat(ds.marginTop) : null,
    interChildGap: lines.length > 1 ? rect(lines[1]).top - rect(lines[0]).bottom : null,
    intendedInterChildGap: ds ? parseFloat(ds.rowGap) : null,
    marker: mr,
    lastDot: dr,
    label: { ...lr, text:label.textContent },
    labelFont: { family: getComputedStyle(label).fontFamily, size: getComputedStyle(label).fontSize },
    lineFont: lines[0] ? { family: getComputedStyle(lines[0]).fontFamily, size: getComputedStyle(lines[0]).fontSize } : null,
    headerGap: lr.left - dr.right,
    intendedGap: parseFloat(getComputedStyle(header).columnGap),
    details: lines.map((line) => ({ ...rect(line), text:line.textContent, clientWidth:line.clientWidth, scrollWidth:line.scrollWidth })),
    documentOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    bodyOverflow: document.body.scrollWidth - document.body.clientWidth,
  };
})()'

set_models() {
  local kind="$1"
  if [[ "$kind" == "short" ]]; then
    payload='["dexgem-responses/gpt-5.6-sol","dexgem-responses/gpt-5.3"]'
  else
    payload='["dexgem-responses/gpt-5.6-sol","anthropic/claude-sonnet-4-5-20250929-xhigh-reasoning"]'
  fi
  printf '%s\n' "$payload" | "${docker[@]}" exec -i "$container" sh -c 'cat > "$HOME/subagent-layout-models.json"'
}

set_state() {
  if [[ "$1" == "idle" ]]; then
    "${docker[@]}" exec "$container" sh -c 'if [ ! -e "$HOME/subagent-layout-idle" ]; then touch "$HOME/subagent-layout-idle"; printf "%s\n" '\''{"type":"message","id":"a2","parentId":"u1","timestamp":"2026-09-11T06:01:00.000Z","message":{"role":"assistant","model":"provider/parent-model","stopReason":"stop","content":[{"type":"text","text":"Parent turn settled while both children remain active."}]}}'\'' >> "$HOME/subagent-layout.jsonl"; fi'
  else
    "${docker[@]}" exec "$container" rm -f /home/tester/subagent-layout-idle
  fi
}

capture() {
  local viewport="$1" width="$2" height="$3" state="$4" models="$5" name
  name="${viewport}-${state}-${models}"
  browser set viewport "$width" "$height" >/dev/null 2>&1 || fail "viewport $name failed"
  set_models "$models"
  set_state "$state"
  # Session-list polling is 5s; this wait also clears the scanner's 2s cache.
  browser wait 6500 >/dev/null 2>&1
  browser eval "$PROBE" --json > "$artifacts/${name}.json" 2>&1 || fail "probe $name failed"
  browser screenshot "$artifacts/${name}.png" >/dev/null 2>&1 || fail "screenshot $name failed"
}

for state in busy idle; do
  for viewport in phone desktop; do
    if [[ "$viewport" == "phone" ]]; then width=390; height=844; else width=1280; height=800; fi
    capture "$viewport" "$width" "$height" "$state" short
    capture "$viewport" "$width" "$height" "$state" long
  done
done

browser errors --json > "$artifacts/browser-errors.json" 2>&1 || true
browser console --json > "$artifacts/browser-console.json" 2>&1 || true
"${docker[@]}" logs "$container" > "$artifacts/server.log" 2>&1 || true

python3 - "$artifacts" <<'PY'
import json
import math
import sys
from pathlib import Path

artifacts = Path(sys.argv[1])

def load(name):
    payload = json.loads((artifacts / f"{name}.json").read_text(encoding="utf-8"))
    assert payload.get("success") is True, (name, payload)
    value = payload["data"]["result"]
    assert isinstance(value, dict), (name, value)
    return value

names = [f"{viewport}-{state}-{models}" for viewport in ("phone", "desktop") for state in ("busy", "idle") for models in ("short", "long")]
probes = {name: load(name) for name in names}
checks = {}
for name, probe in probes.items():
    expected_state = name.split("-")[1]
    checks[f"{name}:state"] = probe["state"] == expected_state
    checks[f"{name}:two_children"] = len(probe["details"]) == 2
    checks[f"{name}:intended_gap"] = math.isclose(probe["headerGap"], probe["intendedGap"], abs_tol=0.25) and math.isclose(probe["intendedGap"], 8.0, abs_tol=0.25)
    checks[f"{name}:bounded_bubble"] = probe["bubble"]["width"] <= probe["availableWidth"] + 0.25
    checks[f"{name}:no_bubble_overflow"] = probe["bubble"]["scrollWidth"] <= probe["bubble"]["clientWidth"]
    checks[f"{name}:no_line_overflow"] = all(line["scrollWidth"] <= line["clientWidth"] for line in probe["details"])
    checks[f"{name}:no_page_overflow"] = probe["documentOverflow"] == 0 and probe["bodyOverflow"] == 0
    label = probe["label"]["text"]
    checks[f"{name}:summary"] = ("subagents: 2" in label) if expected_state == "busy" else label == "▸2 subagents working"
    checks[f"{name}:compact_labels"] = all("tokens:" in line["text"] and "tokens used" not in line["text"] for line in probe["details"])
    # Shared internal layout: matched padding (space-3/space-4), child lines and
    # marker at bubble content-left, tightened header->details gap, modest
    # inter-child gap.
    checks[f"{name}:matched_padding"] = math.isclose(probe["padding"]["top"], 8.0, abs_tol=0.25) and math.isclose(probe["padding"]["left"], 10.0, abs_tol=0.25)
    checks[f"{name}:marker_at_content_left"] = math.isclose(probe["markerOffsetFromContent"], 0.0, abs_tol=0.25)
    checks[f"{name}:details_at_content_left"] = probe["firstLineOffsetFromContent"] is not None and math.isclose(probe["firstLineOffsetFromContent"], 0.0, abs_tol=0.25)
    checks[f"{name}:header_to_details_gap"] = probe["headerToDetailsGap"] is not None and probe["intendedHeaderToDetailsGap"] is not None and math.isclose(probe["headerToDetailsGap"], probe["intendedHeaderToDetailsGap"], abs_tol=0.25) and math.isclose(probe["intendedHeaderToDetailsGap"], 6.0, abs_tol=0.25)
    checks[f"{name}:inter_child_gap"] = probe["interChildGap"] is not None and probe["intendedInterChildGap"] is not None and math.isclose(probe["interChildGap"], probe["intendedInterChildGap"], abs_tol=0.25) and math.isclose(probe["intendedInterChildGap"], 4.0, abs_tol=0.25)
    # Detail lines share the summary's UI font family (narrow exception to the
    # mono-for-data rule, pinned by user preference).
    checks[f"{name}:detail_font_matches_summary"] = probe["lineFont"] is not None and probe["lineFont"]["family"] == probe["labelFont"]["family"]

for viewport in ("phone", "desktop"):
    for state in ("busy", "idle"):
        short = probes[f"{viewport}-{state}-short"]
        long = probes[f"{viewport}-{state}-long"]
        checks[f"{viewport}-{state}:gap_unchanged_by_details"] = math.isclose(short["headerGap"], long["headerGap"], abs_tol=0.25)
        checks[f"{viewport}-{state}:marker_track_unchanged_by_details"] = math.isclose(short["marker"]["width"], long["marker"]["width"], abs_tol=0.25)
        checks[f"{viewport}-{state}:short_shrink_wraps"] = short["bubble"]["width"] < short["availableWidth"] - 20
checks["phone-long:details_wrap"] = all(any(line["height"] > 14.5 for line in probes[f"phone-{state}-long"]["details"]) for state in ("busy", "idle"))
for viewport in ("phone", "desktop"):
    for models in ("short", "long"):
        busy_probe = probes[f"{viewport}-busy-{models}"]
        idle_probe = probes[f"{viewport}-idle-{models}"]
        checks[f"{viewport}-{models}:busy_idle_padding_match"] = math.isclose(busy_probe["padding"]["top"], idle_probe["padding"]["top"], abs_tol=0.25) and math.isclose(busy_probe["padding"]["left"], idle_probe["padding"]["left"], abs_tol=0.25)
        checks[f"{viewport}-{models}:busy_idle_gaps_match"] = math.isclose(busy_probe["headerToDetailsGap"], idle_probe["headerToDetailsGap"], abs_tol=0.25) and math.isclose(busy_probe["interChildGap"], idle_probe["interChildGap"], abs_tol=0.25)

errors_raw = json.loads((artifacts / "browser-errors.json").read_text(encoding="utf-8"))
errors = errors_raw.get("data", {}).get("errors", [])
checks["no_browser_errors"] = errors == []
summary = {"pass": all(checks.values()), "checks": checks, "geometry": probes, "browserErrors": errors}
(artifacts / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
if not summary["pass"]:
    raise SystemExit(1)
PY

(
  cd "$artifacts"
  sha256sum ./*.json ./*.png > SHA256SUMS
)
printf 'PASS: artifacts=%s\n' "$artifacts"
