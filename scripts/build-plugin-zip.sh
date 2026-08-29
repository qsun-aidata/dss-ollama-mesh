#!/usr/bin/env bash
#
# Build the installable Dataiku plugin zip.
#
# `git archive` rather than `zip -r .` is the whole point: this directory also
# holds a virtualenv, caches, and whatever else a developer left lying around,
# and none of that belongs in a plugin anyone installs. Only committed files
# ship, minus the paths marked export-ignore in .gitattributes.
set -euo pipefail

cd "$(dirname "$0")/.."

ref="${1:-HEAD}"
version=$(python3 -c 'import json; print(json.load(open("plugin.json"))["version"])')
out="dist/dss-ollama-mesh-${version}.zip"

if [[ "$ref" == v* && "${ref#v}" != "$version" ]]; then
    echo "error: ref ${ref} does not match plugin version ${version}" >&2
    exit 1
fi

if ! git diff --quiet HEAD -- . 2>/dev/null; then
    echo "warning: working tree has uncommitted changes; building from ${ref}" >&2
fi

mkdir -p dist
rm -f "$out"
git archive --format=zip --output="$out" "$ref"

echo "Built ${out}"
unzip -l "$out"
