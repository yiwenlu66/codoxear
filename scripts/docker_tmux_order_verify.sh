#!/usr/bin/env bash
# Docker-isolated behavioural verifier for the sidebar → tmux window order.
#
# Exercises the real product surface inside the container: a codoxear server,
# a real tmux server, and sessions launched through `POST /api/sessions` with
# `create_in_tmux`. It then asserts that
#   * tmux windows follow the same relative order as `/api/sessions`,
#   * a window codoxear does not own keeps its index,
#   * the viewer keeps the same pane content while its window is renumbered,
#   * snoozing / un-snoozing a session (a real ordering change) is followed, and
#   * the order is stable afterwards.
#
# The application is built from the working tree into an image (nothing is
# imported from the host checkout at run time) and runs under a container-only
# HOME. No host Codoxear runtime, deployment state, tmux socket, or port 8743
# is touched.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/docker_tmux_order_verify.sh [commit]

Build an isolated image from the working tree and verify inside it that tmux
window order follows the sidebar order. Prints the behavioural checks; the
container output is also saved under /tmp/codoxear-docker-tmux-order-results.*

Environment:
  CODOXEAR_TMUX_ORDER_IMAGE     Image tag (default: codoxear-tmux-order:<commit>)
  CODOXEAR_TMUX_ORDER_KEEP       1 to keep the container and build context

The image is always rebuilt from the working tree, so a stale cached tag can
never be verified by mistake.
USAGE
}

case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_ref="${1:-HEAD}"
target_commit="$(git -C "$repo_root" rev-parse --verify "${target_ref}^{commit}")" || {
  echo "unable to resolve commit: $target_ref" >&2
  exit 2
}
short_commit="$(git -C "$repo_root" rev-parse --short=12 "$target_commit")"
image="${CODOXEAR_TMUX_ORDER_IMAGE:-codoxear-tmux-order:${short_commit}}"
container="codoxear-tmux-order-${short_commit}-$$"
root="$(mktemp -d /tmp/codoxear-docker-tmux-order.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-docker-tmux-order-results.XXXXXX)"
readonly root artifacts image container
readonly -a docker=(sudo docker)

cleanup() {
  "${docker[@]}" rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

build_context="$root/build-context"
mkdir -p "$build_context"
git -C "$repo_root" ls-files -z --cached --others --exclude-standard | python3 -c '
import sys
skip = (b".memory/", b"review/", b"build/")
parts = sys.stdin.buffer.read().split(b"\0")
keep = [p for p in parts if p and not p.startswith(skip)]
sys.stdout.buffer.write(b"\0".join(keep) + b"\0")
' > "$root/context-files.nul"
tar --null --no-recursion -C "$repo_root" -T "$root/context-files.nul" -cf "$root/context.tar"
tar -C "$build_context" -xf "$root/context.tar"

# Always rebuild: the image is built from the working tree, so a cached image
# from an identical commit tag would verify stale code.
echo "building $image ..."
"${docker[@]}" build \
  --file "$build_context/docker/tmux-order-verify.Dockerfile" \
  --tag "$image" \
  "$build_context" > "$artifacts/docker-build.log" 2>&1 || {
    echo "docker image build failed; see $artifacts/docker-build.log" >&2
    tail -40 "$artifacts/docker-build.log" >&2 || true
    exit 1
  }

set +e
"${docker[@]}" run --rm --name "$container" --init "$image" 2>&1 | tee "$artifacts/driver.log"
status="${PIPESTATUS[0]}"
set -e

echo
if [[ "$status" -eq 0 ]]; then
  echo "tmux order verification PASSED (artifacts: $artifacts)"
else
  echo "tmux order verification FAILED (status $status; artifacts: $artifacts)" >&2
fi
exit "$status"
