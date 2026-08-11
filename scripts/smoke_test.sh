#!/usr/bin/env bash
# Authenticated browser smoke test for a running Codoxear server.
#
# Usage: scripts/smoke_test.sh [base-url] [password]
#
# The password can instead come from CODOXEAR_SMOKE_PASSWORD or
# CODEX_WEB_PASSWORD. CODOXEAR_SMOKE_ATTEMPTS controls full initial-load plus
# reload attempts (default: 3); CODOXEAR_SMOKE_SESSION names the isolated
# agent-browser session (default: codoxear-smoke).
set -euo pipefail

readonly BASE_URL="${1:-${CODOXEAR_SMOKE_URL:-http://127.0.0.1:8743}}"
readonly SMOKE_PASSWORD="${2:-${CODOXEAR_SMOKE_PASSWORD:-${CODEX_WEB_PASSWORD:-}}}"
readonly SMOKE_SESSION="${CODOXEAR_SMOKE_SESSION:-codoxear-smoke}"
readonly SMOKE_ATTEMPTS="${CODOXEAR_SMOKE_ATTEMPTS:-3}"
readonly SETTLE_MS="${CODOXEAR_SMOKE_SETTLE_MS:-3000}"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/smoke_test.sh [base-url] [password]

Runs an authenticated browser smoke test. It requires at least one rendered
session card, a completed app bootstrap, no recorded JavaScript load error, and
no visible server-contact/load failure caption on the initial load and reload.
EOF
}

if [[ ! "$SMOKE_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "smoke test failed: CODOXEAR_SMOKE_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if [[ -z "$SMOKE_PASSWORD" ]]; then
  usage
  echo "smoke test failed: no password supplied" >&2
  exit 2
fi
if ! command -v agent-browser >/dev/null; then
  echo "smoke test failed: agent-browser is required" >&2
  exit 1
fi

browser() {
  AGENT_BROWSER_SESSION="$SMOKE_SESSION" agent-browser "$@"
}

cleanup() {
  browser close >/dev/null 2>&1 || true
}
trap cleanup EXIT

check_rendered_app() {
  local phase="$1"
  local result
  result="$(browser eval '(() => {
    const bodyText = String(document.body && (document.body.innerText || document.body.textContent) || "").toLowerCase();
    const visibleErrorCaption = ["error: unable to contact server", "codoxear failed to load"]
      .find((caption) => bodyText.includes(caption));
    const sessions = document.querySelector("#sessions");
    const cards = sessions ? sessions.querySelectorAll(":scope > .session").length : 0;
    const sessionListRendered = Boolean(sessions && sessions.dataset.codoxearSessionsRendered === "true");
    const bundleLoaded = Boolean(document.querySelector('script[type="module"][src*="dist/app.bundle.js"]'));
    const checks = {
      noVisibleLoadCaption: !visibleErrorCaption,
      noRecordedLoadError: typeof window.__codoxearLoadError === "undefined",
      appBootstrapped: window.__codoxearAppBootstrapped === true,
      bundleLoaded,
      sessionListRendered,
      sessionCardsRendered: cards > 0,
    };
    const failures = Object.entries(checks).filter(([, passed]) => !passed).map(([name]) => name);
    if (visibleErrorCaption) failures.push(`visible-caption=${visibleErrorCaption}`);
    return failures.length === 0 ? "OK" : `FAIL: ${failures.join(",")}; cards=${cards}`;
  })()' --json 2>&1)" || {
    echo "smoke test failed during ${phase} check: ${result}" >&2
    return 1
  }
  if [[ "$result" != *'"OK"'* ]]; then
    echo "smoke test failed during ${phase} check: ${result}" >&2
    return 1
  fi
}

for attempt in $(seq 1 "$SMOKE_ATTEMPTS"); do
  browser open "$BASE_URL/" >/dev/null 2>&1 || true
  browser fill "#pw" "$SMOKE_PASSWORD" >/dev/null 2>&1 || true
  browser click "#loginBtn" >/dev/null 2>&1 || true
  browser wait "$SETTLE_MS" >/dev/null 2>&1 || true

  if check_rendered_app "initial load (attempt ${attempt})"; then
    browser reload >/dev/null 2>&1 || true
    browser wait "$SETTLE_MS" >/dev/null 2>&1 || true
    if check_rendered_app "reload (attempt ${attempt})"; then
      echo "smoke test passed: initial load and reload rendered a session card"
      exit 0
    fi
  fi

  if (( attempt < SMOKE_ATTEMPTS )); then
    sleep 2
  fi
done

echo "smoke test failed: app did not render cleanly after ${SMOKE_ATTEMPTS} attempt(s)" >&2
exit 1
