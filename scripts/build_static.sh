#!/usr/bin/env bash
# Build the browser application from the ES-module entrypoint.
set -euo pipefail

readonly SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ESBUILD="${ESBUILD:-$SOURCE_ROOT/node_modules/.bin/esbuild}"
readonly ENTRY="$SOURCE_ROOT/codoxear/static/app.js"
readonly OUTPUT="$SOURCE_ROOT/codoxear/static/dist/app.bundle.js"

if [[ ! -x "$ESBUILD" ]]; then
  echo "esbuild executable not found: $ESBUILD (run npm install)" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT")"
"$ESBUILD" "$ENTRY" --bundle --minify --outfile="$OUTPUT" --format=esm
