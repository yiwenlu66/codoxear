#!/usr/bin/env bash
# Deploy a committed Codoxear snapshot without ever serving the source checkout.
#
# Release guards:
# - Node parses every static JavaScript asset, and check_js_refs.py rejects bare
#   function calls in app.js that no local declaration or loaded module provides.
# - The selected/session-index/session-list state declarations are explicit
#   regression tripwires for the renderApp closure.
# - An authenticated browser smoke check requires a completed session-list
#   render (zero or more cards), no application load error, and the controller
#   globals that app.js depends on.
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
if ! command -v node >/dev/null; then
  echo "node is required to syntax-check static JavaScript before deployment" >&2
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

# Parse every exact immutable static asset before changing the installed package
# or restarting the service. Grammar validation cannot detect a bare call whose
# declaration was lost during extraction, so the reference checker examines the
# app shell against its locally declared names and globals registered by scripts
# that index.html actually loads.
while IFS= read -r -d '' javascript_path; do
  node --check "$javascript_path"
done < <(find "$DEPLOY_DIR/codoxear/static" -type f -name '*.js' -print0 | sort -z)
if ! python3 "$DEPLOY_DIR/scripts/check_js_refs.py" "$DEPLOY_DIR/codoxear/static"; then
  echo "app.js reference check failed in deploy snapshot" >&2
  exit 1
fi

# Declaration tripwires cover renderApp state whose absence can otherwise
# surface only after the async session-list render.
for declaration in \
  'let[[:space:]]+latestSessions[[:space:]]*=' \
  'let[[:space:]]+selected[[:space:]]*=' \
  'let[[:space:]]+sessionIndex[[:space:]]*='; do
  if ! grep -Eq "$declaration" "$DEPLOY_DIR/codoxear/static/app.js"; then
    echo "app.js render state declaration check failed: $declaration" >&2
    exit 1
  fi
done

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

# Browser smoke check: syntactic validity and global registration do not prove
# renderApp's closure has every declaration it uses. Authenticate through the
# deployed UI, then use the reusable command to require a visible clean app on
# both its first load and a reload.
if command -v agent-browser >/dev/null && [[ "${CODOXEAR_SKIP_BOOT_CHECK:-0}" != "1" ]]; then
  BOOT_CHECK_PASSWORD="${CODOXEAR_BOOT_CHECK_PASSWORD:-${CODEX_WEB_PASSWORD:-}}"
  if [[ -z "$BOOT_CHECK_PASSWORD" ]]; then
    SERVICE_ENVIRONMENT_FILES="$(systemctl --user show "$SERVICE_NAME" -p EnvironmentFiles --value)"
    BOOT_CHECK_PASSWORD="$(python3 - "$SERVICE_ENVIRONMENT" "$SERVICE_ENVIRONMENT_FILES" <<'PY'
import shlex
import sys
from pathlib import Path


def password_from_fields(fields):
    for field in fields:
        if field.startswith("CODEX_WEB_PASSWORD="):
            return field.split("=", 1)[1]
    return ""


try:
    password = password_from_fields(shlex.split(sys.argv[1]))
except ValueError:
    password = ""
if not password:
    for configured_file in sys.argv[2].splitlines():
        path_text = configured_file.rsplit(" (ignore_errors=", 1)[0].strip()
        if not path_text:
            continue
        try:
            lines = Path(path_text).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                password = password_from_fields(shlex.split(line, comments=True))
            except ValueError:
                continue
            if password:
                break
        if password:
            break
print(password)
PY
)"
  fi
  if [[ -z "$BOOT_CHECK_PASSWORD" ]]; then
    echo "boot check failed: no CODEX_WEB_PASSWORD available for authenticated app smoke test" >&2
    exit 1
  fi

  if ! CODOXEAR_SMOKE_SESSION=deploy-boot CODOXEAR_SMOKE_ATTEMPTS=3 "$SOURCE_ROOT/scripts/smoke_test.sh" "$BASE_URL" "$BOOT_CHECK_PASSWORD"; then
    echo "boot check failed: deployed app did not render a clean session list after authentication" >&2
    exit 1
  fi
fi
printf 'deployed %s from %s\n' "$TARGET_COMMIT" "$DEPLOY_DIR"
