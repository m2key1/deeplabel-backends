#!/bin/bash
# Usage: cluster-image.sh <image:tag> [out.sqsh]
# Turns a built backend image into the squashfs file Slurm's Pyxis/enroot runs
# (MODEL_BACKEND_<NAME>_CLUSTER_IMAGE): the image's root filesystem, with its environment
# in /etc/environment as `enroot import` writes it. Copy the file to the cluster.
set -euo pipefail
image=$1; out=${2:-$(echo "$image" | tr ':/' '--').sqsh}
container=$(docker create "$image")
environment=$(mktemp)
trap 'docker rm -f "$container" >/dev/null; rm -f "$environment"' EXIT
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$image" | sed '/^$/d' > "$environment"
docker cp "$environment" "$container:/etc/environment"
rm -f "$out"
docker export "$container" | sqfstar -quiet -comp zstd "$out"
echo "$out"
