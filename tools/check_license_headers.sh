#!/bin/sh
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Verify that every tracked, non-binary file begins with the AMD copyright
# notice. Shared by the CI workflow (.github/workflows). The copyright text is
# identical across comment styles (/* */, #, @REM, //, <!-- -->), so we match
# the text itself within the first few lines rather than a fixed leader. An
# optional shebang line and/or a UTF-8 BOM are tolerated before the notice.
#
# Usage: check_license_headers.sh
set -eu

# Accepts a single year (2025) or a range (2021-2025).
LICENSE_RE='Copyright \(c\) [0-9]{4}(-[0-9]{4})? Advanced Micro Devices, Inc\. All rights reserved\.'

# How many leading lines may precede the notice (shebang, BOM-only line, etc.).
HEAD_LINES=5

# Coverage: every tracked file, regardless of directory or extension. The
# following are excluded because they cannot or should not carry the notice:
#   * binary files                  — no text to annotate (detected below)
#   * empty / whitespace-only files — nothing to annotate (e.g. placeholders)
#   * *.json                        — JSON has no comment syntax
#   * *.in                          — generated/templated inputs
#   * cmake/Find*.cmake             — third-party CMake find-modules
# (3rdparty/ and build/ are gitignored, so git ls-files never lists them.)
# Collect offenders in a temp file: the loop below runs in a pipeline subshell
# (so a shell variable would not survive), and a file avoids any reliance on
# word-splitting for paths that contain spaces or tabs.
missing="$(mktemp)"
trap 'rm -f "$missing"' EXIT

# Read paths line-by-line with IFS cleared so spaces/tabs in names are kept
# intact (a plain `for f in $(git ls-files)` would word-split them).
git ls-files | while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in
        *.json|*.in|cmake/Find*.cmake) continue ;;
    esac
    # Skip binary files (grep -I flags them via NUL bytes) and empty files.
    LC_ALL=C grep -Iq . "$f" || continue
    # Skip whitespace-only placeholder files.
    [ -n "$(tr -d '[:space:]' < "$f")" ] || continue
    # Skip Git LFS pointer files: without an LFS fetch they hold only metadata,
    # not the real source, so there is no header to verify here.
    case "$(head -n 1 "$f")" in
        "version https://git-lfs.github.com/spec/v1") continue ;;
    esac
    if ! head -n "$HEAD_LINES" "$f" | grep -Eq "$LICENSE_RE"; then
        printf '%s\n' "$f" >> "$missing"
    fi
done

if [ -s "$missing" ]; then
    echo "ERROR: the following files are missing the AMD copyright header:" >&2
    sed 's/^/  /' "$missing" >&2
    echo "" >&2
    echo "Each file must begin (after an optional shebang) with, e.g.:" >&2
    echo "  /* Copyright (c) $(date +%Y) Advanced Micro Devices, Inc. All rights reserved. */   (C/C++)" >&2
    echo "  # Copyright (c) $(date +%Y) Advanced Micro Devices, Inc. All rights reserved.        (scripts, YAML, dotfiles)" >&2
    echo "  <!-- Copyright (c) $(date +%Y) Advanced Micro Devices, Inc. All rights reserved. -->  (Markdown)" >&2
    exit 1
fi

echo "All tracked non-binary files carry the AMD copyright header."
exit 0
