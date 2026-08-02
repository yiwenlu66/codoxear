#!/usr/bin/env bash
# Deploy a committed Codoxear snapshot without ever serving the source checkout.
set -euo pipefail

readonly SERVICE_NAME="codoxear-server.service"
readonly SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly DEPLOY_DIR="${CODOXEAR_DEPLOY_DIR:-$HOME/.local/share/codoxear/deploy}"
readonly UNIT_PATH="${CODOXEAR_SERVICE_UNIT:-$HOME/.config/systemd/user/$SERVICE_NAME}"

usage() {
  cat <<'EOF'
Usage: scripts/deploy.sh <commit-ish>

Creates or updates ~/.local/share/codoxear/deploy as a detached git worktree at
<commit-ish>, installs that snapshot with pipx, points the user service at the
snapshot, restarts only codoxear-server.service, and verifies its HTTP boundary.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

if ! command -v git >/dev/null; then
  echo "git is required" >&2
  exit 1
fi
if ! command -v pipx >/dev/null; then
  echo "pipx is required" >&2
  exit 1
fi
if ! command -v curl >/dev/null; then
  echo "curl is required for the health check" >&2
  exit 1
fi
if [[ ! -f "$UNIT_PATH" ]]; then
  echo "service unit does not exist: $UNIT_PATH" >&2
  exit 1
fi

TARGET_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse --verify "$1^{commit}")" || {
  echo "cannot resolve commit-ish as a commit: $1" >&2
  exit 1
}

update_snapshot() {
  if [[ -e "$DEPLOY_DIR" ]]; then
    if [[ ! -d "$DEPLOY_DIR" ]] || ! git -C "$DEPLOY_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "deploy path exists but is not a git worktree: $DEPLOY_DIR" >&2
      return 1
    fi

    local source_git_dir deploy_git_dir
    source_git_dir="$(cd -- "$SOURCE_ROOT" && cd -- "$(git rev-parse --git-common-dir)" && pwd -P)"
    deploy_git_dir="$(cd -- "$DEPLOY_DIR" && cd -- "$(git rev-parse --git-common-dir)" && pwd -P)"
    if [[ "$source_git_dir" != "$deploy_git_dir" ]]; then
      echo "deploy worktree belongs to a different repository: $DEPLOY_DIR" >&2
      return 1
    fi
    if [[ -n "$(git -C "$DEPLOY_DIR" status --porcelain)" ]]; then
      echo "refusing to replace a dirty deploy worktree: $DEPLOY_DIR" >&2
      return 1
    fi

    git -C "$DEPLOY_DIR" checkout --detach "$TARGET_COMMIT"
  else
    mkdir -p "$(dirname -- "$DEPLOY_DIR")"
    git -C "$SOURCE_ROOT" worktree add --detach "$DEPLOY_DIR" "$TARGET_COMMIT"
  fi

  local actual_commit
  actual_commit="$(git -C "$DEPLOY_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" != "$TARGET_COMMIT" ]]; then
    echo "deploy worktree did not reach requested commit: expected $TARGET_COMMIT, got $actual_commit" >&2
    return 1
  fi
}

# This boundary is deliberate: no pipx or service operation occurs unless the
# detached worktree update completed and is at the requested commit.
update_snapshot || {
  echo "snapshot update failed; service was not reinstalled or restarted" >&2
  exit 1
}

pipx install --force "$DEPLOY_DIR"
PIPX_HOME="$(pipx environment --value PIPX_HOME)"
VENV_PYTHON="$PIPX_HOME/venvs/codoxear/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "pipx install completed but its Codoxear Python is missing: $VENV_PYTHON" >&2
  exit 1
fi

# Preserve every unrelated unit setting (notably Environment/EnvironmentFile)
# while making its import root and invocation use the immutable snapshot.
python3 - "$UNIT_PATH" "$DEPLOY_DIR" "$VENV_PYTHON" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

unit_path = Path(sys.argv[1])
deploy_dir = sys.argv[2]
venv_python = sys.argv[3]
lines = unit_path.read_text().splitlines(keepends=True)
replacements = {
    "WorkingDirectory": deploy_dir,
    "ExecStart": f"{venv_python} -u -m codoxear.server",
}
for directive, value in replacements.items():
    matches = [index for index, line in enumerate(lines) if line.startswith(f"{directive}=")]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {directive}= directive in {unit_path}, found {len(matches)}")
    lines[matches[0]] = f"{directive}={value}\n"

temporary_path = unit_path.with_name(f".{unit_path.name}.{os.getpid()}.tmp")
try:
    temporary_path.write_text("".join(lines))
    os.replace(temporary_path, unit_path)
finally:
    temporary_path.unlink(missing_ok=True)
PY

systemctl --user daemon-reload
systemctl --user restart "$SERVICE_NAME"
systemctl --user is-active --quiet "$SERVICE_NAME"

SERVICE_ENVIRONMENT="$(systemctl --user show "$SERVICE_NAME" -p Environment --value)"
SERVICE_PORT="$(sed -n 's/.*\bCODEX_WEB_PORT=\([^ ]*\).*/\1/p' <<<"$SERVICE_ENVIRONMENT")"
SERVICE_PORT="${SERVICE_PORT:-8743}"
BASE_URL="http://127.0.0.1:$SERVICE_PORT"

wait_for_status() {
  local expected_status="$1"
  local path="$2"
  local status=""
  for _ in {1..30}; do
    status="$(curl --connect-timeout 2 --max-time 5 --silent --output /dev/null --write-out '%{http_code}' "$BASE_URL$path" || true)"
    if [[ "$status" == "$expected_status" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "health check failed for $path: expected $expected_status, got ${status:-connection-error}" >&2
  return 1
}

wait_for_status 200 "/"
wait_for_status 401 "/api/sessions"

# Boot check: the page must not only respond, the app must boot. A broken
# frontend passes the HTTP health check, so assert the load-error surface is
# absent and the core controller globals register before calling this a deploy.
if command -v agent-browser >/dev/null; then
  BOOT_OK=0
  for _ in {1..3}; do
    AGENT_BROWSER_SESSION=deploy-boot agent-browser open "$BASE_URL/" >/dev/null 2>&1 || true
    sleep 3
    BOOT_CHECK="$(AGENT_BROWSER_SESSION=deploy-boot agent-browser eval '(() => { const err = !!document.querySelector("[data-codoxear-load-error]"); const globals = ["CodoxearUrls","CodoxearStorage","CodoxearApi"].every((k) => !!window[k]); return err || !globals ? "FAIL" : "OK"; })()' --json 2>/dev/null || true)"
    AGENT_BROWSER_SESSION=deploy-boot agent-browser close >/dev/null 2>&1 || true
    if [[ "$BOOT_CHECK" == *'"OK"'* ]]; then
      BOOT_OK=1
      break
    fi
    sleep 2
  done
  if [[ "$BOOT_OK" != 1 ]]; then
    echo "boot check failed: the app did not boot cleanly after deploy (load-error surface present or core globals missing)" >&2
    exit 1
  fi
fi
printf 'deployed %s from %s\n' "$TARGET_COMMIT" "$DEPLOY_DIR"
