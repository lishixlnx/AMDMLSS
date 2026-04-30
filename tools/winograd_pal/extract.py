#!/usr/bin/env python3
"""Extract every Winograd PAL relocatable ELF embedded in a `shadersBinReloc.hpp`
into a raw `.o` file under `build/winograd_pal/<Family>/<arch>/<prec>/`.

Each kernel becomes one `.o` file plus an entry in a per-directory
`manifest.json`. The manifest carries the metadata downstream link / re-emit
steps need (original size, source line, header path, bytes-per-line layout)
without forcing them to re-parse the C++ header.

The tool is intentionally pessimistic: any byte-count mismatch, duplicate
symbol or unrecognized layout aborts with a non-zero exit code rather than
silently dropping kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import struct

REPO_ROOT = Path(__file__).resolve().parents[2]
WINOGRAD_ROOT = (
    REPO_ROOT
    / "modules"
    / "shaders"
    / "src"
    / "operators"
    / "impl"
    / "conv"
    / "mxn"
    / "Winograd"
)

# Relative paths (under modules/shaders/src/operators/impl/conv/mxn/Winograd)
# of every shadersBinReloc.hpp we're authoritatively responsible for. We hard-code
# this list so the tool is deterministic; running `extract.py --list` prints
# it out for verification.
SHADER_BIN_RELATIVE_PATHS = (
    "Base/gfx1100/fp16/shadersBinReloc.hpp",
    "Base/gfx1100/fp32/shadersBinReloc.hpp",
    "Base/gfx1201/fp16/shadersBinReloc.hpp",
    "Base/gfx1201/fp32/shadersBinReloc.hpp",
    "Fury/gfx1100/fp16/shadersBinReloc.hpp",
    "Fury/gfx1201/fp16/shadersBinReloc.hpp",
    "Rage/gfx1201/fp16/shadersBinReloc.hpp",
)

KERNEL_HEADER_RE = re.compile(
    r"const\s+StaticShaderType\s*<\s*(?P<size>\d+)\s*>\s+(?P<name>\w+)\s*=\s*\{",
)
M_BINARY_RE = re.compile(r"\.m_binary\s*=\s*\{")
HEX_BYTE_RE = re.compile(r"0x([0-9a-fA-F]{1,2})")


@dataclass
class KernelEntry:
    name: str
    size: int
    source_hpp: str
    source_line: int
    dst_o: str
    sh_info_patched: bool = False


@dataclass
class DirectoryManifest:
    family: str
    arch: str
    precision: str
    source_hpp: str
    kernels: list[KernelEntry]


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Return the index of the `}` that matches the `{` at `open_pos`."""
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


def _decode_bytes_block(block_text: str, expected_size: int, kernel_name: str) -> bytes:
    """Decode a comma-separated `0x..` byte literal block into bytes."""
    matches = HEX_BYTE_RE.findall(block_text)
    decoded = bytes(int(h, 16) for h in matches)
    if len(decoded) != expected_size:
        raise ValueError(
            f"kernel {kernel_name!r}: declared size {expected_size} but "
            f"decoded {len(decoded)} bytes"
        )
    return decoded


def _line_of(text: str, offset: int) -> int:
    """Return the 1-based line number of `offset` within `text`."""
    return text.count("\n", 0, offset) + 1


def _parse_header(hpp_path: Path) -> list[tuple[KernelEntry, bytes]]:
    """Return `(metadata, raw_bytes)` for every kernel in `hpp_path`."""
    text = hpp_path.read_text(encoding="utf-8")

    results: list[tuple[KernelEntry, bytes]] = []
    seen_names: set[str] = set()

    for header_match in KERNEL_HEADER_RE.finditer(text):
        size = int(header_match.group("size"))
        name = header_match.group("name")

        if name in seen_names:
            raise ValueError(
                f"{hpp_path}: duplicate symbol {name!r} (line "
                f"{_line_of(text, header_match.start())})"
            )
        seen_names.add(name)

        outer_open = text.index("{", header_match.end() - 1)
        outer_close = _find_matching_brace(text, outer_open)
        outer_body = text[outer_open + 1 : outer_close]

        binary_match = M_BINARY_RE.search(outer_body)
        if binary_match is None:
            raise ValueError(f"{hpp_path}: kernel {name!r} has no .m_binary block")

        # `.m_binary = {` ends at binary_match.end()-1 (the `{`). The matching
        # `}` is computed in absolute coordinates inside the original text.
        binary_open_abs = outer_open + 1 + binary_match.end() - 1
        binary_close_abs = _find_matching_brace(text, binary_open_abs)
        bytes_block = text[binary_open_abs + 1 : binary_close_abs]

        decoded = _decode_bytes_block(bytes_block, size, name)

        entry = KernelEntry(
            name=name,
            size=size,
            source_hpp=str(hpp_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            source_line=_line_of(text, header_match.start()),
            dst_o="",  # filled in by caller
        )
        results.append((entry, decoded))

    return results


def _sanity_check_first_kernel(decoded: bytes, name: str) -> None:
    """Verify the first 8 bytes look like an AMDGPU PAL relocatable ELF.

    This is purely defensive: if the .hpp ever drifts off-format we want a
    loud failure rather than silently shipping garbage downstream.
    """
    if len(decoded) < 24 or decoded[:4] != b"\x7fELF":
        raise ValueError(f"kernel {name!r}: missing ELF magic")
    if decoded[4] != 0x02:
        raise ValueError(f"kernel {name!r}: not ELFCLASS64 (got {decoded[4]:#04x})")
    if decoded[7] != 0x41:
        raise ValueError(
            f"kernel {name!r}: expected OS/ABI=AMDGPU_PAL (0x41), "
            f"got {decoded[7]:#04x}; this tool only supports PAL ELFs"
        )
    e_type = int.from_bytes(decoded[16:18], "little")
    if e_type != 0x01:
        raise ValueError(
            f"kernel {name!r}: expected e_type=ET_REL (0x01), got {e_type:#06x}"
        )
    e_machine = int.from_bytes(decoded[18:20], "little")
    if e_machine != 0xE0:
        raise ValueError(
            f"kernel {name!r}: expected e_machine=EM_AMDGPU (0xe0), "
            f"got {e_machine:#06x}"
        )


def _patch_symtab_sh_info(data: bytearray, kernel_name: str) -> bool:
    """Repair the `.symtab` section header's `sh_info` field in-place.

    The Winograd PAL pipeline ELFs ship with `sh_info = 0` on `.symtab`,
    which violates the ELF spec (sh_info must equal the index of the first
    non-LOCAL symbol). LLVM's linker rejects these inputs with
    "invalid sh_info in symbol table". This is a one-byte-aligned u32 fix
    that does not alter any symbol or instruction bytes.

    Returns True if the field was modified, False if it was already correct.
    """
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def section_header_offset(index: int) -> int:
        return e_shoff + index * e_shentsize

    def read_section_field(index: int, offset: int, fmt: str):
        return struct.unpack_from(fmt, data, section_header_offset(index) + offset)[0]

    shstr_off = read_section_field(e_shstrndx, 0x18, "<Q")

    def section_name(index: int) -> str:
        name_off = read_section_field(index, 0, "<I")
        end = data.index(b"\x00", shstr_off + name_off)
        return data[shstr_off + name_off : end].decode("ascii")

    symtab_index = next(
        (i for i in range(e_shnum) if section_name(i) == ".symtab"),
        -1,
    )
    if symtab_index < 0:
        raise ValueError(f"kernel {kernel_name!r}: no .symtab section found")

    sym_offset = read_section_field(symtab_index, 0x18, "<Q")
    sym_size = read_section_field(symtab_index, 0x20, "<Q")
    sym_entsize = read_section_field(symtab_index, 0x38, "<Q")
    if sym_entsize == 0:
        raise ValueError(f"kernel {kernel_name!r}: zero symtab entry size")
    symbol_count = sym_size // sym_entsize

    def symbol_binding(index: int) -> int:
        st_info = data[sym_offset + index * sym_entsize + 4]
        return st_info >> 4

    first_non_local = next(
        (i for i in range(symbol_count) if symbol_binding(i) != 0),
        symbol_count,
    )

    sh_info_offset = section_header_offset(symtab_index) + 0x2C
    current = struct.unpack_from("<I", data, sh_info_offset)[0]
    if current == first_non_local:
        return False
    struct.pack_into("<I", data, sh_info_offset, first_non_local)
    return True


def _split_path(rel: str) -> tuple[str, str, str]:
    """Convert `Base/gfx1201/fp16/shadersBinReloc.hpp` -> `(Base, gfx1201, fp16)`."""
    parts = Path(rel).parts
    if len(parts) != 4 or parts[3] != "shadersBinReloc.hpp":
        raise ValueError(f"unexpected shadersBinReloc path layout: {rel!r}")
    return parts[0], parts[1], parts[2]


def extract(out_root: Path, *, clean: bool) -> int:
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total_kernels = 0
    total_bytes = 0

    for rel in SHADER_BIN_RELATIVE_PATHS:
        hpp_path = WINOGRAD_ROOT / rel
        if not hpp_path.is_file():
            print(f"ERROR: missing source header: {hpp_path}", file=sys.stderr)
            return 1

        family, arch, precision = _split_path(rel)
        out_dir = out_root / family / arch / precision
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            parsed = _parse_header(hpp_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if not parsed:
            print(f"ERROR: no kernels found in {hpp_path}", file=sys.stderr)
            return 1

        kernel_entries: list[KernelEntry] = []
        for entry, decoded in parsed:
            try:
                _sanity_check_first_kernel(decoded, entry.name)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

            patched = bytearray(decoded)
            try:
                entry.sh_info_patched = _patch_symtab_sh_info(patched, entry.name)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

            dst_o = out_dir / f"{entry.name}.o"
            dst_o.write_bytes(bytes(patched))
            entry.dst_o = str(dst_o.relative_to(out_root)).replace("\\", "/")
            kernel_entries.append(entry)
            total_kernels += 1
            total_bytes += len(decoded)

        manifest = DirectoryManifest(
            family=family,
            arch=arch,
            precision=precision,
            source_hpp=str(hpp_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            kernels=kernel_entries,
        )
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            f"  {family}/{arch}/{precision}: {len(kernel_entries)} kernel(s), "
            f"{sum(e.size for e in kernel_entries):,} bytes"
        )

    print()
    print(
        f"Extracted {total_kernels} kernel(s) totalling {total_bytes:,} bytes "
        f"into {out_root}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "build" / "winograd_pal",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="do not wipe the output directory before extracting",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the source headers this tool will process and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for rel in SHADER_BIN_RELATIVE_PATHS:
            print(WINOGRAD_ROOT.joinpath(rel).as_posix())
        return 0
    return extract(args.out, clean=not args.no_clean)


if __name__ == "__main__":
    sys.exit(main())
