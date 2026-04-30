#!/usr/bin/env python3
"""Rewrite the standalone-data .hpp files under ``archive/`` so that they
depend only on the C++ standard library.

Transformations performed for each ``*.hpp`` under ``archive/``:

* ``StaticShaderType<N> NAME = { .m_binary = { ... }, .m_shaderType = ... };``
  becomes ``inline constexpr auto NAME = std::to_array<std::uint8_t>({ ... });``
  (all metadata fields are dropped; they referenced project-internal types).

* ``const unsigned char NAME[N] = { ... };`` becomes the same stdlib form.

* ``const UINT32 NAME[] = { ... };`` becomes
  ``inline constexpr auto NAME = std::to_array<std::uint32_t>({ ... });``.

* ``inline const TrainedDecisionTree<N, K> NAME { ints, ints, floats, ints,
  {{int, ...}, ...} };`` is preserved by also emitting a stdlib-only
  ``TrainedDecisionTree`` definition local to the file (the C 2D array is
  rewritten as ``std::array<std::array<std::int32_t, K>, N>``).

* The enclosing namespace is rewritten to ``archive::<dirs>`` derived from
  the file's path under ``archive/``, e.g. ``archive/conv/mxn/Winograd/Base/
  gfx1100/fp16/shadersBinReloc.hpp`` becomes
  ``namespace archive::conv::mxn::winograd::base::gfx1100::fp16``.

The script is intentionally conservative: any file it cannot fully rewrite
is reported and left untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = REPO_ROOT / "archive"

# Files that cannot be made standalone with the standard library only.
SKIP_BASENAMES = {"shadersUtils.hpp", "shadersUtils.cpp"}

COPYRIGHT_LINE = "/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */"

# Some directory names map to identifiers that would otherwise start with a
# digit (which is illegal as a C++ namespace name) or that we want to keep
# consistent with how the live operator code names them. lowercase() handles
# CamelCase entries (e.g. "Winograd" -> "winograd") automatically.
NAMESPACE_LOWER_OVERRIDES: dict[str, str] = {
    "1x1": "one_by_one",
    "1xn": "one_by_n",
    "16x16": "tile_16x16",
    "32": "q32",
    "64x64": "tile_64x64",
    "128": "q128",
    "mxn": "mxn",
}

# ----------------------------------------------------------------------------
# Path -> namespace
# ----------------------------------------------------------------------------


def namespace_for(path: Path) -> str:
    relative = path.relative_to(ARCHIVE_DIR).parent
    parts = ["archive"]
    for p in relative.parts:
        token = NAMESPACE_LOWER_OVERRIDES.get(p, p.lower())
        parts.append(token)
    return "::".join(parts)


# ----------------------------------------------------------------------------
# Hex byte tidying
# ----------------------------------------------------------------------------

BYTES_PER_LINE = 16
HEX_BYTE = re.compile(r"0x[0-9a-fA-F]+")


def reformat_uint8_bytes(blob: str) -> str:
    """Reformat a raw byte body as 16 columns of 0xNN tokens, indented."""

    bytes_ = HEX_BYTE.findall(blob)
    lines: list[str] = []
    for i in range(0, len(bytes_), BYTES_PER_LINE):
        chunk = bytes_[i : i + BYTES_PER_LINE]
        lines.append("    " + ",".join(f"0x{int(b, 16):02x}" for b in chunk))
    return ",\n".join(lines)


HEX_OR_DEC_TOKEN = re.compile(r"0x[0-9a-fA-F]+|-?\d+")


def reformat_uint32_words(blob: str) -> str:
    tokens = HEX_OR_DEC_TOKEN.findall(blob)
    formatted: list[str] = []
    for tok in tokens:
        if tok.startswith(("0x", "0X")):
            formatted.append(f"0x{int(tok, 16):08x}")
        else:
            formatted.append(tok)
    columns = 8
    lines: list[str] = []
    for i in range(0, len(formatted), columns):
        chunk = formatted[i : i + columns]
        lines.append("    " + ",".join(chunk))
    return ",\n".join(lines)


# ----------------------------------------------------------------------------
# Per-file statistics (populated as the rewriters run)
# ----------------------------------------------------------------------------


@dataclass
class FileStats:
    path: Path
    bytes_in: int = 0
    bytes_out: int = 0
    namespace: str = ""
    static_shader_blocks: int = 0
    unsigned_char_blocks: int = 0
    uint32_blocks: int = 0
    tree_blocks: int = 0
    binary_blob_aliases: int = 0
    mlssarg_replacements: int = 0
    mlss_enum_replacements: int = 0
    includes_stripped: list[str] = field(default_factory=list)
    includes_emitted: list[str] = field(default_factory=list)
    leftover_tokens: list[str] = field(default_factory=list)


@dataclass
class Totals:
    files_processed: int = 0
    files_rewritten: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    static_shader_blocks: int = 0
    unsigned_char_blocks: int = 0
    uint32_blocks: int = 0
    tree_blocks: int = 0
    binary_blob_aliases: int = 0
    mlssarg_replacements: int = 0
    mlss_enum_replacements: int = 0

    def absorb(self, stats: FileStats) -> None:
        self.bytes_in += stats.bytes_in
        self.bytes_out += stats.bytes_out
        self.static_shader_blocks += stats.static_shader_blocks
        self.unsigned_char_blocks += stats.unsigned_char_blocks
        self.uint32_blocks += stats.uint32_blocks
        self.tree_blocks += stats.tree_blocks
        self.binary_blob_aliases += stats.binary_blob_aliases
        self.mlssarg_replacements += stats.mlssarg_replacements
        self.mlss_enum_replacements += stats.mlss_enum_replacements


# ----------------------------------------------------------------------------
# Pattern-specific rewriters
# ----------------------------------------------------------------------------

STATIC_SHADER_RE = re.compile(
    r"""
    (?:inline\s+)?
    const\s+StaticShaderType\s*<\s*\d+\s*>\s+(?P<name>[A-Za-z_]\w*)\s*=\s*\{
    \s*
    (?:.*?\bm_binary\b\s*=\s*\{)
    (?P<bytes>.*?)
    \}\s*,?
    (?:[^{}]*?)
    \}\s*;
    """,
    re.DOTALL | re.VERBOSE,
)

UNSIGNED_CHAR_RE = re.compile(
    r"""
    const\s+unsigned\s+char\s+(?P<name>[A-Za-z_]\w*)
    \s*\[\s*\d+\s*\]\s*=\s*\{
    (?P<bytes>.*?)
    \}\s*;
    """,
    re.DOTALL | re.VERBOSE,
)

UINT32_ARRAY_RE = re.compile(
    r"""
    const\s+UINT32\s+(?P<name>[A-Za-z_]\w*)
    \s*\[\s*\]\s*=\s*\{
    (?P<words>.*?)
    \}\s*;
    """,
    re.DOTALL | re.VERBOSE,
)

# Aliases of the form
#   constexpr BinaryBlob NAME = BINARYBLOB_SETTER(SOURCE);
# left over from a previous partial conversion of archive/gemm/mxn/hip/. The
# BinaryBlob struct/macro come from the live ``shaders/`` headers; we only
# want a stdlib reference to the existing byte array here.
BINARY_BLOB_ALIAS_RE = re.compile(
    r"""
    (?:inline\s+)?constexpr\s+BinaryBlob\s+
    (?P<name>[A-Za-z_]\w*)\s*=\s*
    BINARYBLOB_SETTER\s*\(\s*(?P<source>[A-Za-z_]\w*)\s*\)\s*;
    """,
    re.VERBOSE,
)

NAMESPACE_OPEN_RE = re.compile(r"namespace\s+[A-Za-z_][\w:]*\s*\{")
NAMESPACE_CLOSE_RE = re.compile(r"\}\s*//\s*namespace[^\n]*\n?")
INCLUDE_LINE_RE = re.compile(r'^[ \t]*#\s*include[ \t]*[<"][^>"]+[>"][ \t]*\n', re.MULTILINE)
PRAGMA_ONCE_RE = re.compile(r"^[ \t]*#\s*pragma\s+once[ \t]*\n", re.MULTILINE)
COPYRIGHT_RE = re.compile(r"/\*[^*]*Copyright[^*]*\*/\s*\n", re.IGNORECASE)
LEADING_BLANK_RE = re.compile(r"^\s+", re.MULTILINE)

# ----------------------------------------------------------------------------
# TrainedDecisionTree (shadersConstants.hpp)
# ----------------------------------------------------------------------------

TREE_HEADER_RE = re.compile(
    r"inline\s+const\s+TrainedDecisionTree\s*<\s*(?P<n>\d+)\s*,\s*(?P<k>\d+)\s*>\s+(?P<name>[A-Za-z_]\w*)\s*\{",
    re.DOTALL,
)


def slice_balanced(text: str, start: int) -> tuple[str, int]:
    """Return the substring from ``start`` (which must be ``{``) up to and
    including the matching ``}``, plus the index after the closing brace."""

    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        i += 1
    raise ValueError("unbalanced braces")


INTEGER_TOKEN = re.compile(r"-?\d+")
FLOAT_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def format_int_array(values: list[str], elem_type: str) -> str:
    columns = 16
    lines = []
    for i in range(0, len(values), columns):
        chunk = values[i : i + columns]
        lines.append("    " + ",".join(chunk))
    body = ",\n".join(lines)
    return f"std::array<{elem_type}, {len(values)}>{{{{\n{body}\n}}}}"


def format_classes(rows: list[list[str]], k: int) -> str:
    inner: list[str] = []
    for row in rows:
        inner.append(
            f"        std::array<std::int32_t, {k}>{{{{ " + ", ".join(row) + " }}"
        )
    body = ",\n".join(inner)
    return (
        f"std::array<std::array<std::int32_t, {k}>, {len(rows)}>{{{{\n"
        f"{body}\n"
        "}}"
    )


def parse_int_brace(text: str) -> list[str]:
    return INTEGER_TOKEN.findall(text)


def parse_float_brace(text: str) -> list[str]:
    raw = FLOAT_TOKEN.findall(text)
    out: list[str] = []
    for tok in raw:
        if "." in tok or "e" in tok or "E" in tok:
            out.append(tok + "f")
        else:
            out.append(tok + ".0f")
    return out


def parse_classes_brace(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    i = text.find("{")
    while i != -1 and i < len(text):
        end = text.find("}", i)
        if end == -1:
            break
        row_text = text[i + 1 : end]
        rows.append(INTEGER_TOKEN.findall(row_text))
        i = text.find("{", end + 1)
    return rows


def rewrite_tree_block(body: str, n: int, k: int, name: str) -> str:
    fields = []
    cursor = 0
    for _ in range(4):
        open_idx = body.find("{", cursor)
        slice_text, cursor = slice_balanced(body, open_idx)
        fields.append(slice_text)
    classes_open = body.find("{", cursor)
    classes_text, _ = slice_balanced(body, classes_open)

    left = parse_int_brace(fields[0])
    right = parse_int_brace(fields[1])
    thresh = parse_float_brace(fields[2])
    indices = parse_int_brace(fields[3])
    classes = parse_classes_brace(classes_text)

    if not (len(left) == len(right) == len(thresh) == len(indices) == n):
        raise ValueError(f"tree {name}: array length mismatch (expected {n})")
    if len(classes) != n:
        raise ValueError(f"tree {name}: classes row count {len(classes)} != {n}")
    for row in classes:
        if len(row) != k:
            raise ValueError(f"tree {name}: class row width {len(row)} != {k}")

    return (
        f"inline constexpr TrainedDecisionTree<{n}, {k}> {name}\n{{\n"
        + "    "
        + format_int_array(left, "std::int32_t")
        + ",\n    "
        + format_int_array(right, "std::int32_t")
        + ",\n    "
        + format_int_array(thresh, "float")
        + ",\n    "
        + format_int_array(indices, "std::int32_t")
        + ",\n    "
        + format_classes(classes, k)
        + "\n};\n"
    )


def rewrite_constants_file(text: str, stats: FileStats) -> str:
    out: list[str] = []
    cursor = 0
    for match in TREE_HEADER_RE.finditer(text):
        out.append(text[cursor : match.start()])
        n = int(match.group("n"))
        k = int(match.group("k"))
        name = match.group("name")
        body_open = match.end() - 1
        body_text, after = slice_balanced(text, body_open)
        out.append(rewrite_tree_block(body_text[1:-1], n, k, name))
        stats.tree_blocks += 1
        # consume the trailing ';' if present
        if after < len(text) and text[after] == ";":
            after += 1
        cursor = after
    out.append(text[cursor:])
    return "".join(out)


# ----------------------------------------------------------------------------
# File rewriter
# ----------------------------------------------------------------------------


def strip_includes_and_pragmas(text: str, stats: FileStats) -> str:
    stats.includes_stripped = [
        m.group(0).strip() for m in INCLUDE_LINE_RE.finditer(text)
    ]
    text = COPYRIGHT_RE.sub("", text, count=1)
    text = INCLUDE_LINE_RE.sub("", text)
    text = PRAGMA_ONCE_RE.sub("", text)
    return text


def strip_namespace_braces(text: str) -> str:
    text = NAMESPACE_OPEN_RE.sub("", text, count=1)
    text = NAMESPACE_CLOSE_RE.sub("", text)
    # also strip a bare trailing "}" that may close the namespace if there's
    # no "// namespace X" comment
    return text


def rewrite_static_shader(match: re.Match, stats: FileStats) -> str:
    stats.static_shader_blocks += 1
    name = match.group("name")
    bytes_text = match.group("bytes")
    body = reformat_uint8_bytes(bytes_text)
    return f"inline constexpr auto {name} = std::to_array<std::uint8_t>({{\n{body}\n}});"


def rewrite_unsigned_char(match: re.Match, stats: FileStats) -> str:
    stats.unsigned_char_blocks += 1
    name = match.group("name")
    bytes_text = match.group("bytes")
    body = reformat_uint8_bytes(bytes_text)
    return f"inline constexpr auto {name} = std::to_array<std::uint8_t>({{\n{body}\n}});"


def rewrite_uint32_array(match: re.Match, stats: FileStats) -> str:
    stats.uint32_blocks += 1
    name = match.group("name")
    words_text = match.group("words")
    body = reformat_uint32_words(words_text)
    return f"inline constexpr auto {name} = std::to_array<std::uint32_t>({{\n{body}\n}});"


def rewrite_binary_blob_alias(match: re.Match, stats: FileStats) -> str:
    stats.binary_blob_aliases += 1
    name = match.group("name")
    source = match.group("source")
    return f"inline constexpr auto& {name} = {source};"


def required_includes(body: str, is_constants: bool) -> list[str]:
    includes: list[str] = ["<array>", "<cstdint>"]
    # The shadersConstants helper struct exposes a std::string_view name field;
    # bring the header in unconditionally for those files.
    if is_constants:
        includes.append("<string_view>")
    return includes


TREE_HELPER_BLOCK = """\
template <std::size_t N, std::size_t NumClasses>
struct TrainedDecisionTree
{
    std::array<std::int32_t, N> m_leftChildren;
    std::array<std::int32_t, N> m_rightChildren;
    std::array<float, N> m_thresholds;
    std::array<std::int32_t, N> m_indices;
    std::array<std::array<std::int32_t, NumClasses>, N> m_classes;
};

// Stdlib-only mirror of the project's MLSSarg/MLSSenum metadata. The integer
// values match the original ``X << 16`` encoding used in the live API headers
// so that downstream code can compare them directly if needed.
enum class KernelArgType : std::uint32_t
{
    None    =  0u << 16u,
    Bool    =  1u << 16u,
    Int8    =  2u << 16u,
    Uint8   =  3u << 16u,
    Int16   =  4u << 16u,
    Uint16  =  5u << 16u,
    Int32   =  8u << 16u,
    Uint32  =  9u << 16u,
    Int64   = 10u << 16u,
    Uint64  = 11u << 16u,
    Float16 = 16u << 16u,
    Float32 = 17u << 16u,
    Float64 = 18u << 16u,
};

struct KernelArg
{
    std::int32_t     m_place;
    KernelArgType    m_type;
    bool             m_isPointer;
    std::uint32_t    m_indirectionLevel;
    bool             m_isConst;
    bool             m_isInput;
    bool             m_isReturn;
    std::string_view m_name;
};

"""

ARG_TYPE_REPLACEMENTS = {
    "MLSS_BOOL":    "KernelArgType::Bool",
    "MLSS_INT8":    "KernelArgType::Int8",
    "MLSS_UINT8":   "KernelArgType::Uint8",
    "MLSS_INT16":   "KernelArgType::Int16",
    "MLSS_UINT16":  "KernelArgType::Uint16",
    "MLSS_INT32":   "KernelArgType::Int32",
    "MLSS_UINT32":  "KernelArgType::Uint32",
    "MLSS_INT64":   "KernelArgType::Int64",
    "MLSS_UINT64":  "KernelArgType::Uint64",
    "MLSS_FLOAT16": "KernelArgType::Float16",
    "MLSS_FLOAT32": "KernelArgType::Float32",
    "MLSS_FLOAT64": "KernelArgType::Float64",
}


def replace_mlss_arg_metadata(text: str, stats: FileStats) -> str:
    text, count = re.subn(r"\bMLSSarg\b", "KernelArg", text)
    stats.mlssarg_replacements = count
    enum_total = 0
    for old, new in ARG_TYPE_REPLACEMENTS.items():
        text, count = re.subn(rf"\b{old}\b", new, text)
        enum_total += count
    stats.mlss_enum_replacements = enum_total
    return text


def rewrite_file(path: Path) -> tuple[bool, str, FileStats]:
    """Return ``(changed, message, stats)``."""

    raw = path.read_text(encoding="utf-8")
    is_constants = path.name == "shadersConstants.hpp"
    is_il = path.name == "shadersIL.hpp"

    stats = FileStats(path=path, bytes_in=len(raw.encode("utf-8")))

    text = raw
    text = strip_includes_and_pragmas(text, stats)
    text = strip_namespace_braces(text)

    if is_constants:
        text = rewrite_constants_file(text, stats)
        text = replace_mlss_arg_metadata(text, stats)
    else:
        if is_il:
            text = UINT32_ARRAY_RE.sub(lambda m: rewrite_uint32_array(m, stats), text)
        text = STATIC_SHADER_RE.sub(lambda m: rewrite_static_shader(m, stats), text)
        text = UNSIGNED_CHAR_RE.sub(lambda m: rewrite_unsigned_char(m, stats), text)
        text = BINARY_BLOB_ALIAS_RE.sub(
            lambda m: rewrite_binary_blob_alias(m, stats), text
        )

    # Drop now-orphan "} // namespace ..." comments and any stray closing
    # brace at end of file.
    text = re.sub(r"\}\s*//[^\n]*namespace[^\n]*\n?", "", text)
    text = text.rstrip()
    if text.endswith("}"):
        text = text[:-1].rstrip()

    # Sanity check: nothing project-specific should remain.
    leftovers = [
        token
        for token in (
            "StaticShaderType",
            "ShaderTypesFlags",
            "make_shader_descriptor",
            "make_binary_blob",
            "Binaries::",
            "MetaCmdCaps",
            "MLSSarg",
            "MLSSenum",
            "MLSSint",
            "MLSSuint",
            "MLSSbool",
            "MLSSconststring",
            "UINT32",
            "BinaryBlob",
            "BINARYBLOB_SETTER",
        )
        if re.search(rf"\b{re.escape(token)}\b", text)
    ]
    if leftovers:
        stats.leftover_tokens = leftovers
        return False, f"unhandled tokens: {', '.join(leftovers)}", stats

    namespace = namespace_for(path)
    stats.namespace = namespace
    include_list = required_includes(text, is_constants)
    stats.includes_emitted = list(include_list)
    includes = "\n".join(f"#include {inc}" for inc in include_list)

    helper = TREE_HELPER_BLOCK if is_constants else ""

    body = text.strip()
    new_text = (
        f"{COPYRIGHT_LINE}\n"
        "#pragma once\n\n"
        f"{includes}\n\n"
        f"namespace {namespace}\n"
        "{\n\n"
        f"{helper}"
        f"{body}\n\n"
        f"}} // namespace {namespace}\n"
    )

    stats.bytes_out = len(new_text.encode("utf-8"))

    if new_text == raw:
        return False, "no change", stats
    path.write_text(new_text, encoding="utf-8", newline="\n")
    return True, "rewritten", stats


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def candidate_files() -> Iterable[Path]:
    for path in sorted(ARCHIVE_DIR.rglob("*.hpp")):
        if path.name in SKIP_BASENAMES:
            continue
        yield path


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KiB", "MiB", "GiB"):
        value /= 1024.0
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
    return f"{n} B"


def print_file_report(stats: FileStats, changed: bool, message: str) -> None:
    rel = stats.path.relative_to(REPO_ROOT)
    header = "OK   " if changed else "skip "
    print(f"{header} {rel}  ({message})")
    print(f"        namespace      : {stats.namespace or '-'}")
    print(
        f"        bytes          : in={format_bytes(stats.bytes_in)}, "
        f"out={format_bytes(stats.bytes_out)} "
        f"(delta={stats.bytes_out - stats.bytes_in:+d} B)"
    )
    print(
        f"        rewrites       : "
        f"static_shader={stats.static_shader_blocks}, "
        f"unsigned_char={stats.unsigned_char_blocks}, "
        f"uint32_arrays={stats.uint32_blocks}, "
        f"trees={stats.tree_blocks}, "
        f"binary_blob_aliases={stats.binary_blob_aliases}"
    )
    if stats.mlssarg_replacements or stats.mlss_enum_replacements:
        print(
            f"        mlss metadata  : "
            f"MLSSarg->KernelArg={stats.mlssarg_replacements}, "
            f"enum tokens={stats.mlss_enum_replacements}"
        )
    if stats.includes_stripped:
        print(f"        stripped incs  : {', '.join(stats.includes_stripped)}")
    if stats.includes_emitted:
        print(f"        emitted incs   : {', '.join(stats.includes_emitted)}")
    if stats.leftover_tokens:
        print(f"        LEFTOVERS      : {', '.join(stats.leftover_tokens)}")


def print_summary(totals: Totals) -> None:
    print("\n=== summary ===")
    print(f"  files processed   : {totals.files_processed}")
    print(f"  files rewritten   : {totals.files_rewritten}")
    print(f"  files unchanged   : {totals.files_skipped}")
    print(f"  files failed      : {totals.files_failed}")
    print(
        f"  bytes             : in={format_bytes(totals.bytes_in)}, "
        f"out={format_bytes(totals.bytes_out)} "
        f"(delta={totals.bytes_out - totals.bytes_in:+d} B)"
    )
    print(
        f"  rewrites totals   : "
        f"static_shader={totals.static_shader_blocks}, "
        f"unsigned_char={totals.unsigned_char_blocks}, "
        f"uint32_arrays={totals.uint32_blocks}, "
        f"trees={totals.tree_blocks}, "
        f"binary_blob_aliases={totals.binary_blob_aliases}"
    )
    print(
        f"  mlss metadata     : "
        f"MLSSarg->KernelArg={totals.mlssarg_replacements}, "
        f"enum tokens={totals.mlss_enum_replacements}"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change without writing files.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PATH",
        help="restrict rewriting to a specific file (relative to the repo).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print per-file statistics (namespace, byte counts, rewrite counts, "
        "stripped/emitted includes, leftovers).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print an aggregated summary at the end of the run.",
    )
    args = parser.parse_args(argv)

    files: list[Path]
    if args.only:
        files = [REPO_ROOT / p for p in args.only]
    else:
        files = list(candidate_files())

    failures: list[tuple[Path, str]] = []
    rewrote = 0
    skipped = 0
    totals = Totals()

    for path in files:
        if args.check:
            print(f"would process {path.relative_to(REPO_ROOT)}")
            continue
        try:
            changed, message, stats = rewrite_file(path)
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {path.relative_to(REPO_ROOT)}  {exc}")
            totals.files_failed += 1
            continue

        totals.files_processed += 1
        if changed:
            rewrote += 1
            totals.files_rewritten += 1
        else:
            skipped += 1
            totals.files_skipped += 1
        totals.absorb(stats)

        if args.verbose:
            print_file_report(stats, changed, message)
        else:
            flag = "OK   " if changed else "skip "
            print(f"{flag} {path.relative_to(REPO_ROOT)}  ({message})")

    print(f"\nrewrote {rewrote}/{len(files)} files; {len(failures)} failures")
    if args.summary or args.verbose:
        print_summary(totals)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
