#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Re-emit each Winograd `shadersBinReloc.hpp` as a sibling
`shadersBinNonReloc.hpp` whose kernel bodies are the linked (`ET_DYN`,
OS/ABI=AMDGPU_PAL) outputs produced by `link.{ps1,sh}`.

For every kernel in the source header we keep:
  * the surrounding copyright header, includes, namespace, and comments;
  * the kernel's leading `// ...` comments;
  * the trailing field initializers (m_kernelName, m_compilerVersion,
    m_codeObjectVersion, m_shaderType).

We change exactly four things per kernel:
  1. the templated size in `StaticShaderType<SIZE>`,
  2. the symbol name (suffixed with `_NonReloc`),
  3. the `.m_binary = { ... }` byte block,
  4. `.m_isRelocatable = true` -> `.m_isRelocatable = false`.

Before touching any linked outputs the script also asserts that our byte
encoder round-trips one of the *original* relocatables exactly, so encoder
bugs are caught before they reach disk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

KERNEL_HEADER_RE = re.compile(
    r"const\s+StaticShaderType\s*<\s*(?P<size>\d+)\s*>\s+(?P<name>\w+)\s*=\s*\{",
)
M_BINARY_RE = re.compile(r"\.m_binary\s*=\s*\{")
TEMPLATE_SIZE_RE = re.compile(r"StaticShaderType\s*<\s*\d+\s*>")
HEX_BYTE_RE = re.compile(r"0x([0-9a-fA-F]{1,2})")
IS_RELOCATABLE_TRUE_RE = re.compile(r"\.m_isRelocatable\s*=\s*true")

BYTES_PER_LINE = 20
LINE_INDENT = "    "
NONRELOC_SUFFIX = "_NonReloc"


@dataclass
class Edit:
    start: int
    end: int
    text: str


def _find_matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unterminated brace starting at offset {open_pos}")


def _encode_bytes_block(data: bytes) -> str:
    """Render a byte sequence in the same row-of-20 style as shadersBinReloc.hpp."""
    lines: list[str] = []
    for chunk_start in range(0, len(data), BYTES_PER_LINE):
        chunk = data[chunk_start : chunk_start + BYTES_PER_LINE]
        formatted = ",".join(f"0x{b:02x}" for b in chunk)
        # Every emitted row ends with a trailing comma so the final byte and
        # the closing brace follow the same comma-terminated convention as
        # the source headers.
        lines.append(f"{LINE_INDENT}{formatted},")
    return "\n" + "\n".join(lines) + "\n" + LINE_INDENT


def _decode_bytes_block(block_text: str) -> bytes:
    return bytes(int(h, 16) for h in HEX_BYTE_RE.findall(block_text))


def _self_test_encoder(source_hpp: Path) -> None:
    """Round-trip the first kernel of `source_hpp` through our encoder.

    We decode the original byte block, re-encode it with `_encode_bytes_block`,
    decode the encoded text again, and verify the two byte sequences match.
    Catches any silent corruption in the encoder before we touch real outputs.
    """
    text = source_hpp.read_text(encoding="utf-8")
    header = KERNEL_HEADER_RE.search(text)
    if header is None:
        raise RuntimeError(f"{source_hpp}: no kernel headers found, cannot self-test")
    outer_open = text.index("{", header.end() - 1)
    outer_close = _find_matching_brace(text, outer_open)
    body = text[outer_open + 1 : outer_close]
    bin_match = M_BINARY_RE.search(body)
    if bin_match is None:
        raise RuntimeError(f"{source_hpp}: first kernel has no .m_binary block")
    bin_open_abs = outer_open + 1 + bin_match.end() - 1
    bin_close_abs = _find_matching_brace(text, bin_open_abs)
    raw = _decode_bytes_block(text[bin_open_abs + 1 : bin_close_abs])

    encoded = _encode_bytes_block(raw)
    redecoded = _decode_bytes_block(encoded)
    if redecoded != raw:
        raise RuntimeError(
            f"{source_hpp}: encoder round-trip failed "
            f"({len(raw)} bytes in, {len(redecoded)} bytes out)"
        )


@dataclass
class KernelLayout:
    name: str
    size: int
    header_start: int          # start of `const StaticShaderType...`
    header_end: int            # end of the matched header (just after `=\s*{`)
    outer_open: int            # absolute offset of outer `{`
    outer_close: int           # absolute offset of matching outer `}`
    binary_open: int           # absolute offset of `.m_binary = {`'s `{`
    binary_close: int          # absolute offset of matching `}`


def _layout_kernels(text: str) -> list[KernelLayout]:
    layouts: list[KernelLayout] = []
    for m in KERNEL_HEADER_RE.finditer(text):
        outer_open = text.index("{", m.end() - 1)
        outer_close = _find_matching_brace(text, outer_open)
        body = text[outer_open + 1 : outer_close]
        bin_match = M_BINARY_RE.search(body)
        if bin_match is None:
            raise ValueError(f"kernel {m.group('name')!r} has no .m_binary block")
        bin_open = outer_open + 1 + bin_match.end() - 1
        bin_close = _find_matching_brace(text, bin_open)
        layouts.append(
            KernelLayout(
                name=m.group("name"),
                size=int(m.group("size")),
                header_start=m.start(),
                header_end=m.end(),
                outer_open=outer_open,
                outer_close=outer_close,
                binary_open=bin_open,
                binary_close=bin_close,
            )
        )
    return layouts


def _apply_edits(text: str, edits: list[Edit]) -> str:
    """Apply non-overlapping edits in reverse order so offsets stay valid."""
    for e in sorted(edits, key=lambda x: x.start, reverse=True):
        text = text[: e.start] + e.text + text[e.end :]
    return text


def _build_edits_for_kernel(
    text: str, layout: KernelLayout, linked_bytes: bytes
) -> list[Edit]:
    edits: list[Edit] = []

    # 1. Templated size: rewrite only within the kernel header span.
    header_text = text[layout.header_start : layout.header_end]
    new_header = TEMPLATE_SIZE_RE.sub(
        f"StaticShaderType<{len(linked_bytes)}>", header_text, count=1
    )
    # 2. Symbol name suffix: insert `_NonReloc` after the original name.
    new_header = re.sub(
        rf"\b{re.escape(layout.name)}\b",
        layout.name + NONRELOC_SUFFIX,
        new_header,
        count=1,
    )
    if new_header != header_text:
        edits.append(Edit(layout.header_start, layout.header_end, new_header))

    # 3. .m_binary block: replace the brace contents (exclusive of the braces).
    edits.append(
        Edit(
            layout.binary_open + 1,
            layout.binary_close,
            _encode_bytes_block(linked_bytes),
        )
    )

    # 4. .m_isRelocatable = true  ->  ... = false (only inside this kernel).
    body_start = layout.outer_open + 1
    body_end = layout.outer_close
    body = text[body_start:body_end]
    new_body, n = IS_RELOCATABLE_TRUE_RE.subn(".m_isRelocatable = false", body, count=1)
    if n != 1:
        raise ValueError(
            f"kernel {layout.name!r}: expected exactly one "
            f"`.m_isRelocatable = true` initializer, found {n}"
        )
    # Re-anchor the edit at absolute coordinates by locating the substring.
    rel_offset = body.index("true")
    abs_offset = body_start + rel_offset
    edits.append(Edit(abs_offset, abs_offset + len("true"), "false"))

    return edits


def reemit(build_root: Path) -> int:
    if not build_root.is_dir():
        print(f"ERROR: build root does not exist: {build_root}", file=sys.stderr)
        return 1

    manifests = sorted(build_root.rglob("manifest.json"))
    if not manifests:
        print(
            f"ERROR: no manifest.json found under {build_root}; run extract.py + link first",
            file=sys.stderr,
        )
        return 1

    total_kernels = 0
    written_files: list[Path] = []

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hpp = REPO_ROOT / manifest["source_hpp"]
        if not source_hpp.is_file():
            print(f"ERROR: missing source header: {source_hpp}", file=sys.stderr)
            return 1

        # Encoder self-test on the first kernel of this header (catches
        # encoder regressions before we touch the linked outputs).
        try:
            _self_test_encoder(source_hpp)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        text = source_hpp.read_text(encoding="utf-8")
        try:
            layouts = _layout_kernels(text)
        except ValueError as exc:
            print(f"ERROR: {source_hpp}: {exc}", file=sys.stderr)
            return 1

        layouts_by_name = {layout.name: layout for layout in layouts}
        manifest_dir = manifest_path.parent

        all_edits: list[Edit] = []
        for kernel_meta in manifest["kernels"]:
            name = kernel_meta["name"]
            if name not in layouts_by_name:
                print(
                    f"ERROR: {source_hpp}: kernel {name!r} listed in manifest but "
                    "not present in source header",
                    file=sys.stderr,
                )
                return 1

            linked_path = manifest_dir / f"{name}.linked.elf"
            if not linked_path.is_file():
                print(
                    f"ERROR: missing linked ELF for {name}: {linked_path}\n"
                    f"  Did link.ps1/link.sh run?",
                    file=sys.stderr,
                )
                return 1
            linked_bytes = linked_path.read_bytes()
            if not linked_bytes:
                print(f"ERROR: {linked_path}: empty file", file=sys.stderr)
                return 1

            all_edits.extend(
                _build_edits_for_kernel(text, layouts_by_name[name], linked_bytes)
            )
            total_kernels += 1

        new_text = _apply_edits(text, all_edits)
        out_path = source_hpp.with_name("shadersBinNonReloc.hpp")
        out_path.write_text(new_text, encoding="utf-8")
        written_files.append(out_path)

        print(
            f"  {manifest['family']}/{manifest['arch']}/{manifest['precision']}: "
            f"emitted {out_path.relative_to(REPO_ROOT)} "
            f"({len(manifest['kernels'])} kernel(s))"
        )

    print()
    print(f"Wrote {len(written_files)} header(s) covering {total_kernels} kernel(s).")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--build-root",
        type=Path,
        default=REPO_ROOT / "build" / "winograd_pal",
        help="directory containing extract+link outputs (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return reemit(args.build_root)


if __name__ == "__main__":
    sys.exit(main())
