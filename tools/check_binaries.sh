#!/bin/sh
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Fail if any added/modified file that Git treats as binary lives outside the
# directories permitted to contain binaries. Shared by the pre-commit hook
# (.githooks/pre-commit) and the CI workflow (.github/workflows).
#
# Usage:
#   check_binaries.sh --staged              Inspect staged changes (pre-commit).
#   check_binaries.sh --diff <git-args...>  Inspect a diff range, e.g.
#                                           --diff BASE...HEAD  or  --diff BASE HEAD
#
# A file is considered binary when `git diff --numstat` reports "-" for both the
# added and deleted line counts (Git's own binary detection, honouring
# .gitattributes). Only Added/Modified entries are checked so pre-existing
# binaries and renames do not trip the gate.
set -eu

# Directories allowed to contain binaries. Keep in sync with .gitattributes.
ALLOWED_PREFIXES="archive/ shaders_drop/"

usage() {
    echo "usage: $0 --staged | --diff <git-diff-args...>" >&2
    exit 2
}

mode=${1:-}
[ -n "$mode" ] || usage
shift

case "$mode" in
    --staged)
        numstat=$(git diff --cached --numstat --diff-filter=AM --)
        ;;
    --diff)
        [ "$#" -ge 1 ] || usage
        numstat=$(git diff --numstat --diff-filter=AM "$@" --)
        ;;
    *)
        usage
        ;;
esac

# Build an awk guard from the allowed prefixes so the matching logic has a single
# source of truth.
offenders=$(
    printf '%s\n' "$numstat" | awk -v allowed="$ALLOWED_PREFIXES" '
        BEGIN { n = split(allowed, prefixes, " ") }
        # numstat columns are tab-separated: <added> <deleted> <path>
        $1 == "-" && $2 == "-" {
            path = $3
            for (i = 1; i <= n; i++) {
                if (prefixes[i] != "" && index(path, prefixes[i]) == 1) next
            }
            print path
        }
    ' FS='\t'
)

if [ -n "$offenders" ]; then
    echo "ERROR: binary files are only allowed under: $ALLOWED_PREFIXES" >&2
    echo "The following binary files are outside those directories:" >&2
    # $offenders is newline-delimited; print it as-is (an unquoted expansion
    # here would word-split paths that contain spaces).
    printf '%s\n' "$offenders" | sed 's/^/  /' >&2
    echo "" >&2
    echo "Move them under archive/ or shaders_drop/, or commit a text representation instead." >&2
    exit 1
fi

exit 0
