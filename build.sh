#!/bin/bash
# Usage: build.sh <wrapper> [clone of its model repository] [tag]
# Builds <wrapper>-backend:<tag> from this repository. A wrapper with a MODEL_REVISION is
# built with its model code at that revision, taken from the local clone - nothing is
# fetched from it; one without (its model comes from PyPI) needs no clone.
set -euo pipefail
wrapper=$1; here=$(cd "$(dirname "$0")" && pwd); image=${wrapper//_/-}-backend
commit=$(cd "$here" && git rev-parse --short HEAD 2>/dev/null || echo dev)
context=()
if [ -f "$here/$wrapper/MODEL_REVISION" ]; then
  clone=$2; tag_arg=${3:-}
  revision=$(sed -n 's/^revision=//p' "$here/$wrapper/MODEL_REVISION")
  tag=${tag_arg:-$commit-${revision:0:7}}
  model=$(mktemp -d)
  trap 'rm -rf "$model"' EXIT
  git -C "$clone" archive "$revision" | tar -x -C "$model"
  context=(--build-context model="$model")
else
  tag=${2:-$commit}
fi
docker build -f "$here/$wrapper/Dockerfile" "${context[@]}" -t "$image:$tag" "$here"
echo "$image:$tag"
