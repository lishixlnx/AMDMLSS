#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# End-to-end orchestrator: extract -> link -> reemit.
#
# Runs:
#   1. python tools/winograd_pal/extract.py
#   2. tools/winograd_pal/link.sh
#   3. python tools/winograd_pal/reemit.py
#
# Cleans <repo>/build/winograd_pal first (idempotent re-runs). Any step
# returning a non-zero exit code stops the pipeline.
#
# Usage:
#   tools/winograd_pal/run.sh [--build-root DIR] [--lld PATH]
#                             [--readelf PATH] [--only KERNEL]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

build_root="${repo_root}/build/winograd_pal"
lld=""
readelf=""
only_kernel=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-root) build_root="$2"; shift 2 ;;
        --lld)        lld="$2";        shift 2 ;;
        --readelf)    readelf="$2";    shift 2 ;;
        --only)       only_kernel="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

echo "==> winograd_pal pipeline"
echo "    repo root: ${repo_root}"
echo "    build:     ${build_root}"
echo

echo "==> [1/3] extract"
python "${script_dir}/extract.py" --out "${build_root}"
echo

echo "==> [2/3] link"
link_args=(--build-root "${build_root}")
[[ -n "${lld}" ]]         && link_args+=(--lld "${lld}")
[[ -n "${readelf}" ]]     && link_args+=(--readelf "${readelf}")
[[ -n "${only_kernel}" ]] && link_args+=(--only "${only_kernel}")
bash "${script_dir}/link.sh" "${link_args[@]}"
echo

if [[ -n "${only_kernel}" ]]; then
    echo "==> [3/3] reemit SKIPPED (--only set; per-directory output would be incomplete)"
    exit 0
fi

echo "==> [3/3] reemit"
python "${script_dir}/reemit.py" --build-root "${build_root}"
echo
echo "==> done"
