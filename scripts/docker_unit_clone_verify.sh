#!/usr/bin/env bash
# Run committed tests in a verified Docker clone, never the image's old sources.
# Usage: CODOXEAR_UNIT_VERIFY_IMAGE=<git/Python/pytest/tinycss2/Node/esbuild image> \
#   scripts/docker_unit_clone_verify.sh [local-branch] [pytest arguments...]
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
branch="${1:-upload-robustness-integration}"
[[ $# == 0 ]] || shift
[[ $# != 0 ]] || set -- -q
image="${CODOXEAR_UNIT_VERIFY_IMAGE:?set an image containing git, Python, pytest, tinycss2, Node and esbuild}"
ref="refs/heads/$branch"
git -C "$repo" show-ref --verify --quiet "$ref" || { echo "local branch required: $branch" >&2; exit 2; }
expected="$(git -C "$repo" rev-parse "$ref")"
root="$(mktemp -d /tmp/codoxear-unit-clone-build.XXXXXX)"
artifacts="$(mktemp -d /tmp/codoxear-unit-clone-results.XXXXXX)"
container="codoxear-unit-clone-$$"
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
  /bin/rm -rf "$root"
  echo "artifacts=$artifacts"
}
trap cleanup EXIT
printf '%s\n' "$expected" > "$artifacts/expected-head.txt"
printf '%s\n' "$branch" > "$artifacts/branch.txt"
docker image inspect "$image" --format '{{.Id}}' > "$artifacts/image-id.txt"
# A raw SHA may advertise no refs and produce an empty bundle. Bundle the real
# branch and explicitly select the same advertised branch when cloning.
git -C "$repo" bundle create "$root/repo.bundle" "$ref"
git -C "$repo" bundle list-heads "$root/repo.bundle" | tee "$artifacts/bundle-heads.txt"
docker run --rm --name "$container" \
  -v "$root/repo.bundle:/input/repo.bundle:ro" \
  "$image" sh -c '
    set -eu
    branch="$1"; expected="$2"; shift 2
    git clone --quiet --branch "$branch" /input/repo.bundle /tmp/codoxear-suite
    cd /tmp/codoxear-suite
    actual="$(git rev-parse HEAD)"
    test "$actual" = "$expected" || { echo "wrong clone HEAD: $actual != $expected" >&2; exit 2; }
    export PYTHONPATH="$PWD:$PWD/tests"
    export PYTHONDONTWRITEBYTECODE=1
    python3 scripts/docker_clone_provenance.py "$expected"
    python3 -m pytest "$@"
  ' verify "$branch" "$expected" "$@" 2>&1 | tee "$artifacts/output.log"
