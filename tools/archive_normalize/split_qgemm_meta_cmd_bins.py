#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Move archive/qgemm metaCmdBin headers into the mxn/hlsl/wmma tree.

Each legacy metaCmdBin*.h file contains one or more DXIL UINT32 arrays. The
destination layout is derived from the symbol name:

    qgemm/mxn/hlsl/wmma/<NZQ|WZQ>/<ratio>/<tile>/<gfx>/<dtype>/shadersIL.hpp
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
QGEMM_DIR = REPO_ROOT / "archive" / "qgemm"

COPYRIGHT_LINE = "/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */"

SYMBOL_RE = re.compile(
    r"^const\s+UINT32\s+"
    r"(?P<symbol>QGemm_WMMA_(?P<tile>16x16|64x64)_Gfx(?P<gfx>11|12)_FP16_W_"
    r"(?P<weight>U?INT4)_IlBinary_b(?P<ratio>32|128)_(?P<z>WZ|NZ))"
    r"\s*\[\]\s*=\s*\{",
    re.MULTILINE,
)


def collect_comment_block(text: str, start: int) -> str:
    lines = text[:start].splitlines()
    comment_lines: list[str] = []

    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            comment_lines.append(line)
            continue
        if not stripped and comment_lines:
            continue
        break

    comment_lines.reverse()
    return "\n".join(comment_lines).rstrip()


def find_array_end(text: str, body_start: int) -> int:
    marker = "\n};"
    end = text.find(marker, body_start)
    if end == -1:
        raise ValueError("array terminator not found")
    return end + len(marker)


def destination_for(match: re.Match[str]) -> Path:
    z_dir = f"{match.group('z')}Q"
    dtype = "fp16_u4" if match.group("weight") == "UINT4" else "fp16_s4"
    arch = "gfx1100" if match.group("gfx") == "11" else "gfx1201"

    return (
        QGEMM_DIR
        / "mxn"
        / "hlsl"
        / "wmma"
        / z_dir
        / match.group("ratio")
        / match.group("tile")
        / arch
        / dtype
        / "shadersIL.hpp"
    )


def write_shader_file(source_text: str, match: re.Match[str]) -> Path:
    start = match.start()
    end = find_array_end(source_text, match.end())
    comment = collect_comment_block(source_text, start)
    array_block = source_text[start:end].rstrip()
    dest = destination_for(match)

    body_parts = [
        COPYRIGHT_LINE,
        "#pragma once",
        "",
        '#include "dxc.h"',
        "",
    ]
    if comment:
        body_parts.append(comment)
    body_parts.append(array_block)
    new_text = "\n".join(body_parts) + "\n"

    if dest.exists():
        old_text = dest.read_text(encoding="utf-8")
        if old_text != new_text:
            raise FileExistsError(f"{dest.relative_to(REPO_ROOT)} already exists with different content")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(new_text, encoding="utf-8", newline="\n")
    return dest


def main() -> int:
    sources = sorted(QGEMM_DIR.glob("metaCmdBin*.h"))
    if not sources:
        print("No metaCmdBin*.h files found.")
        return 0

    written: list[Path] = []
    removed: list[Path] = []

    for source in sources:
        text = source.read_text(encoding="utf-8")
        matches = list(SYMBOL_RE.finditer(text))
        if not matches:
            raise ValueError(f"No recognised QGemm WMMA arrays in {source.relative_to(REPO_ROOT)}")

        for match in matches:
            dest = write_shader_file(text, match)
            written.append(dest)
            print(f"{source.relative_to(REPO_ROOT)} :: {match.group('symbol')} -> {dest.relative_to(REPO_ROOT)}")

        source.unlink()
        removed.append(source)

    print(f"\nCreated/verified {len(written)} shadersIL.hpp file(s)")
    print(f"Removed {len(removed)} legacy metaCmdBin header(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
