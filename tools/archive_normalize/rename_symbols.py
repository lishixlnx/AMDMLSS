#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Rename C++ shader variables under ``archive/`` to the AMD GPU spec form
``<operation>_<dtype>_<variant>_<arch>``.

Pipeline
========

1. ``--generate-mapping`` (read-only): scans every ``archive/*.hpp``, derives
   a best-guess target name per symbol, writes ``rename_mapping.json``.

2. default mode: reads the (possibly user-edited) ``rename_mapping.json`` and

   * verifies the mapping is consistent (uniqueness within each file, no
     unresolved cross-file references inside ``archive/``);
   * rewrites each ``.hpp`` in place using whole-word substitution;
   * removes any ``inline constexpr auto& X = Y;`` alias declaration that
     would become self-referential after renaming;
   * updates ``tools/archive_normalize/verify_compile.cpp`` so its sentinel
     symbol references stay in sync;
   * dumps a ``rename_mapping.applied.json`` audit trail.

Both modes accept ``--verbose`` and ``--summary`` for diagnostics.
ELF payloads (the byte arrays themselves) and ``KernelArg::m_name`` strings
are never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = REPO_ROOT / "archive"
TOOL_DIR = Path(__file__).resolve().parent
MAPPING_PATH = TOOL_DIR / "rename_mapping.json"
APPLIED_PATH = TOOL_DIR / "rename_mapping.applied.json"
VERIFY_CPP_PATH = TOOL_DIR / "verify_compile.cpp"

SKIP_BASENAMES = {"shadersUtils.hpp", "shadersUtils.cpp"}

# Files whose only top-level symbols are non-kernel data (decision trees,
# kernel-arg metadata). Renaming would not match the spec convention because
# these are not GPU kernel binaries.
NON_KERNEL_FILES = {
    ARCHIVE_DIR / "conv" / "1x1" / "wmma" / "shadersConstants.hpp",
}

# (path-prefix relative to archive/, canonical amdgpu.* operation token).
# Longest prefix wins.
OP_BY_DIR_PREFIX = [
    ("qgemm/mxn/hlsl",        "qgemm"),
    ("conv/1x1/wmma",        "gemm_add"),
    ("conv/1x1",             "gemm_add"),
    ("conv/dilated",         "conv_relu_add"),
    ("conv/dw/hlsl",         "conv_relu_add"),
    ("conv/mxn/Misa",        "conv_relu_add"),
    ("conv/mxn/Winograd",    "conv_relu_add"),
    ("gemm/1xn/hlsl",        "gemm_add"),
    ("gemm/mxn/ck",          "gemm_add"),
    ("gemm/mxn/hip",         "gemm_add"),
    ("gqa/wmma",             "gqaattn"),
    ("mha/wmma",             "gqaattn"),
]

# Path segments that should be retained as variant tokens (most-meaningful
# kernel categorisation derived from directory layout). "ck" / "hip" are
# build-flavours, not kernel variants, so they are intentionally excluded.
PATH_FAMILY_HINTS = {
    "wmma":     "wmma",
    "misa":     "misa",
    "winograd": "winograd",
    "fury":     "fury",
    "rage":     "rage",
    "base":     "base",
    "hlsl":     "hlsl",
    "no_wmma":  "no_wmma",
    "nzq":      "nzq",
    "wzq":      "wzq",
    "32":       "q32",
    "128":      "q128",
    "16x16":    "16x16",
    "64x64":    "64x64",
    "dw":       "dw",
    "dilated":  "dilated",
    "mha":      "mha",
}

# Tokens to drop unconditionally.
NOISE_TOKENS = {
    "elf", "ilbinary", "il", "binary", "bin", "coba",
    "hip", "amdgcn", "amdhsa", "amd",
    "q", "w", "int4", "uint4",
    "forward",
    "attention", "attn",
    "rocwmma", "roc",
}

# Tokens to drop because the canonical operation already encodes them.
OP_REDUNDANT = {
    "qgemm":            {"qgemm", "gemm", "quantized"},
    "gemm_add":         {"gemm", "gemm2d", "conv"},
    "gemm_gemm_add":    {"gemm", "gemm2d", "conv"},
    "conv_relu_add":    {"conv", "gemm", "gemm2d"},
    "conv_relu_mul":    {"conv", "gemm", "gemm2d"},
    "trans_conv_trans": {"conv", "transpose"},
    "gqaattn":          {"gqa", "gqaattn"},
    "rdb":              set(),
    "QKVProj":          set(),
}

DTYPE_TOKENS_SHORT = {
    "f16", "f32", "f64", "fp16", "fp32", "fp64", "bf16", "int8",
    "fp16_s4", "fp16_u4",
}

CANONICAL_TOKEN = {
    "b32": "q32",
    "b128": "q128",
    "nz": "nzq",
    "wz": "wzq",
}

ARCH_PATTERN = re.compile(r"^gfx\d+$")

# Truncated arch tokens (gfx11/gfx12) that legacy symbols use; mapped to the
# set of full arch names they may refer to (used when deciding to drop them).
TRUNCATED_GFX_FOR_PATH = {
    "gfx1100": {"gfx11"},
    "gfx1101": {"gfx11"},
    "gfx1102": {"gfx11"},
    "gfx1150": {"gfx11"},
    "gfx1151": {"gfx11"},
    "gfx1200": {"gfx12"},
    "gfx1201": {"gfx12"},
}

# Marketing codenames -> canonical arch (used to drop them when they match the
# path). Codenames that DO NOT match the path are kept as a variant token to
# preserve kernel differentiation (e.g. Fury/gfx1100 contains both Navi31 and
# Navi33 variants).
NAVI_TO_ARCH = {
    "navi31": "gfx1100",
    "navi32": "gfx1101",
    "navi33": "gfx1102",
    "navi41": "gfx1200",
    "navi48": "gfx1201",
}

TILE_PATTERN = re.compile(r"^\d+x\d+(?:x\d+)?$")
LAYOUT_TOKENS = {"nn", "nt", "tn", "tt"}
STRIDE_PATTERN = re.compile(r"^stride\d.*$")
DOT_PATTERN = re.compile(r"^dot\d+$")
NAVI_PATTERN = re.compile(r"^navi\d+$")

# Canonical sort priorities. Lower = earlier in the variant slot.
PREFIX_RANK = {
    "packed_kv": 0, "packed_qkv": 0, "unpacked": 0,
    "mha":       1,
    "cross":     2, "self":      2,
    "fallback":  3,
    "wmma":      4, "no_wmma":   4, "hlsl":      4, "misa":      4,
    "winograd":  4, "fury":      4, "rage":      4, "base": 4,
    "nzq":       5, "wzq":       5, "q32":       5.5, "q128": 5.5,
    "dw":        5, "dilated":   5,
}

SYMBOL_RE = re.compile(
    r"^inline\s+constexpr\s+(?:auto&?\s+|TrainedDecisionTree<[^>]+>\s+)([A-Za-z_]\w*)",
    re.MULTILINE,
)

ALIAS_RE = re.compile(
    r"^[ \t]*inline\s+constexpr\s+auto&\s+(?P<lhs>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<rhs>[A-Za-z_]\w*)\s*;\s*\n?",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def split_camel(s: str) -> str:
    """Insert underscores at CamelCase / digit boundaries, then lowercase."""

    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)        # camel: aA -> a_A
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)  # ABc -> A_Bc
    s = re.sub(r"(\d)([A-Z])", r"\1_\2", s)           # 1A -> 1_A
    return s.lower()


def split_known_op_prefix(s: str) -> str:
    """Insert underscores between known op stems and an immediately following
    digit so e.g. ``gemm2d128x128x16`` tokenises into ``gemm2d`` + ``128x...``.

    The ``gemm`` alternative uses a negative lookahead so we never split the
    middle of ``gemm2d`` (which would yield a stray ``2d`` token).
    """

    pattern = re.compile(
        r"(?:gemm(?!2d)|gemm2d|conv|misa|fury|rage|winograd)(?=\d)",
        re.IGNORECASE,
    )
    s = pattern.sub(lambda m: m.group(0) + "_", s)
    # Special: ``1xN`` (digit, x, capital N) followed by an uppercase letter
    # is the legacy ``GEMM 1xN`` shape with a single trailing layout suffix
    # (e.g. ``1xNC`` / ``1xNR``). Keep ``1xn`` as one token, split the suffix.
    s = re.sub(r"1xN(?=[A-Z])", "1xn_", s)
    return s


def fold_known_phrases(tokens: list[str]) -> list[str]:
    """Glue back tokens that should remain a single word: ``packed_kv`` /
    ``packed_qkv``, tile geometries (``1`` ``x`` ``1`` -> ``1x1``), stride
    descriptors, and the ``roc wmma`` library tag."""

    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        # roc + wmma -> drop both (handled in noise filter via combined token)
        if tok == "roc" and i + 1 < n and tokens[i + 1] == "wmma":
            out.append("rocwmma")
            i += 2
            continue
        # no + wmma -> compound token
        if tok == "no" and i + 1 < n and tokens[i + 1] == "wmma":
            out.append("no_wmma")
            i += 2
            continue
        # packed + kv|qkv
        if tok == "packed" and i + 1 < n and tokens[i + 1] in ("kv", "qkv"):
            out.append(f"packed_{tokens[i + 1]}")
            i += 2
            continue
        # N + x + M ( + x + K)
        if (
            i + 2 < n
            and tok.isdigit()
            and tokens[i + 1] == "x"
            and tokens[i + 2].isdigit()
        ):
            tile = f"{tok}x{tokens[i + 1]}{tokens[i + 2]}"
            j = i + 3
            if j + 1 < n and tokens[j] == "x" and tokens[j + 1].isdigit():
                tile += f"x{tokens[j + 1]}"
                j += 2
            out.append(tile)
            i = j
            continue
        # stride + N [+ dec|dil]
        if tok == "stride" and i + 1 < n and tokens[i + 1].isdigit():
            piece = f"stride{tokens[i + 1]}"
            j = i + 2
            if j < n and tokens[j] in ("dec", "dil"):
                piece += tokens[j]
                j += 1
            out.append(piece)
            i = j
            continue
        # already-folded "stride1" / "stride2" followed by dec|dil (the
        # camel-case splitter separates "Stride2Dec" into "stride2" + "dec").
        if tok.startswith("stride") and tok[6:].isdigit() and \
                i + 1 < n and tokens[i + 1] in ("dec", "dil"):
            out.append(tok + tokens[i + 1])
            i += 2
            continue
        # dot + N
        if tok == "dot" and i + 1 < n and tokens[i + 1].isdigit():
            out.append(f"dot{tokens[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def tokenize(name: str) -> list[str]:
    s = split_known_op_prefix(name)
    s = split_camel(s)
    parts = [p for p in re.split(r"_+", s) if p]
    return fold_known_phrases(parts)


# ---------------------------------------------------------------------------
# Path -> op / dtype / arch / family hints
# ---------------------------------------------------------------------------


def derive_op(rel_path: Path) -> str:
    rel = rel_path.parent.as_posix()
    best: tuple[int, str] = (-1, "unknown")
    for prefix, op in OP_BY_DIR_PREFIX:
        if rel == prefix or rel.startswith(prefix + "/"):
            if len(prefix) > best[0]:
                best = (len(prefix), op)
    return best[1]


def derive_dtype(rel_path: Path) -> str | None:
    for part in rel_path.parts:
        lower = part.lower()
        if lower in {"fp16", "fp32", "fp64", "bf16", "int8", "fp16_s4", "fp16_u4"}:
            return lower
    return None


def derive_arch(rel_path: Path) -> str | None:
    for part in rel_path.parts:
        if ARCH_PATTERN.match(part.lower()):
            return part.lower()
    return None


def derive_family_tokens(rel_path: Path) -> list[str]:
    out: list[str] = []
    for part in rel_path.parts:
        lower = part.lower()
        if lower in PATH_FAMILY_HINTS:
            out.append(PATH_FAMILY_HINTS[lower])
    return out


# ---------------------------------------------------------------------------
# Filtering & canonical sort
# ---------------------------------------------------------------------------


def filter_tokens(
    tokens: list[str],
    op: str,
    dtype: str | None,
    arch: str | None,
) -> list[str]:
    out: list[str] = []
    truncated = TRUNCATED_GFX_FOR_PATH.get(arch or "", set())
    for t in tokens:
        t = CANONICAL_TOKEN.get(t, t)
        if t in NOISE_TOKENS:
            continue
        if t in OP_REDUNDANT.get(op, set()):
            continue
        if t in DTYPE_TOKENS_SHORT or (dtype and t == dtype):
            continue
        if dtype and "_" in dtype and t in dtype.split("_"):
            continue
        if arch and t == arch:
            continue
        if arch and t in truncated:
            continue
        if t in NAVI_TO_ARCH and NAVI_TO_ARCH[t] == arch:
            continue
        out.append(t)
    return out


def rank_token(token: str, idx: int) -> tuple[float, int, str]:
    if token in PREFIX_RANK:
        return (PREFIX_RANK[token], idx, token)
    if NAVI_PATTERN.match(token):
        return (4.5, idx, token)
    if TILE_PATTERN.match(token):
        return (6.0, idx, token)
    if token in LAYOUT_TOKENS:
        return (8.0, idx, token)
    if STRIDE_PATTERN.match(token):
        return (9.0, idx, token)
    if DOT_PATTERN.match(token):
        return (10.0, idx, token)
    return (7.0, idx, token)


def canonical_sort(tokens: list[str]) -> list[str]:
    return [t for _, _, t in sorted(rank_token(t, i) for i, t in enumerate(tokens))]


def dedupe_preserve_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Per-symbol rename
# ---------------------------------------------------------------------------


@dataclass
class SymbolRename:
    file: Path
    old: str
    new: str
    op: str
    dtype: str | None
    arch: str | None
    variant_tokens: list[str]
    notes: list[str] = field(default_factory=list)


def derive_new_name(symbol: str, file_path: Path) -> SymbolRename:
    rel = file_path.relative_to(ARCHIVE_DIR)
    op = derive_op(rel)
    dtype = derive_dtype(rel)
    arch = derive_arch(rel)
    family = derive_family_tokens(rel)

    raw_tokens = tokenize(symbol)
    filtered = filter_tokens(raw_tokens, op, dtype, arch)

    # Inject family hints (preserving any already in the symbol).
    combined = dedupe_preserve_order(family + filtered)

    variant = canonical_sort(combined)

    # If there is no architecture (HLSL IL), treat the family hint that would
    # otherwise lead the variant slot as the trailing "platform" tag instead,
    # so the symbol still has a stable suffix.
    trailing = None
    if arch is None and "hlsl" in variant:
        variant = [t for t in variant if t != "hlsl"]
        trailing = "hlsl"

    parts: list[str] = [op]
    if dtype:
        parts.append(dtype)
    parts.extend(variant)
    if arch:
        parts.append(arch)
    elif trailing:
        parts.append(trailing)

    new_name = "_".join(parts)

    notes: list[str] = []
    if op == "unknown":
        notes.append("UNKNOWN_OP")
    if not re.match(r"^[A-Za-z_]\w*$", new_name):
        notes.append("INVALID_IDENTIFIER")
    suspect = [t for t in variant if not re.match(r"^[a-z0-9_]+$", t)]
    if suspect:
        notes.append("SUSPECT_TOKEN:" + ",".join(suspect))

    return SymbolRename(
        file=file_path,
        old=symbol,
        new=new_name,
        op=op,
        dtype=dtype,
        arch=arch,
        variant_tokens=variant,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# File enumeration & symbol scan
# ---------------------------------------------------------------------------


def candidate_files() -> list[Path]:
    files: list[Path] = []
    for p in sorted(ARCHIVE_DIR.rglob("*.hpp")):
        if p.name in SKIP_BASENAMES:
            continue
        files.append(p)
    return files


def scan_symbols(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8")
    return [m.group(1) for m in SYMBOL_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Mapping generation
# ---------------------------------------------------------------------------


def generate_mapping(verbose: bool, summary: bool) -> int:
    entries: list[dict] = []
    op_counter: Counter[str] = Counter()
    files_seen = 0
    suspect_total = 0

    for fp in candidate_files():
        files_seen += 1
        if fp in NON_KERNEL_FILES:
            if verbose:
                print(f"SKIP {fp.relative_to(REPO_ROOT)}  (non-kernel symbols)")
            continue
        symbols = scan_symbols(fp)
        if not symbols:
            if verbose:
                print(f"SKIP {fp.relative_to(REPO_ROOT)}  (no symbols)")
            continue

        if verbose:
            print(f"FILE {fp.relative_to(REPO_ROOT)}  ({len(symbols)} symbol(s))")

        for sym in symbols:
            r = derive_new_name(sym, fp)
            op_counter[r.op] += 1
            if any(n.startswith("SUSPECT_TOKEN") or n in ("UNKNOWN_OP",
                                                          "INVALID_IDENTIFIER")
                   for n in r.notes):
                suspect_total += 1
            entries.append({
                "file":  fp.relative_to(REPO_ROOT).as_posix(),
                "old":   r.old,
                "new":   r.new,
                "op":    r.op,
                "dtype": r.dtype,
                "arch":  r.arch,
                "notes": r.notes,
            })
            if verbose:
                marker = "  (no change)" if r.old == r.new else ""
                note = f"  [{','.join(r.notes)}]" if r.notes else ""
                print(f"        {sym}\n          -> {r.new}{marker}{note}")

    MAPPING_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {MAPPING_PATH.relative_to(REPO_ROOT)}: "
          f"{len(entries)} entries from {files_seen} files")

    if summary or verbose:
        print("\n=== summary ===")
        print(f"  files scanned    : {files_seen}")
        print(f"  symbols total    : {len(entries)}")
        print(f"  symbols suspect  : {suspect_total}")
        for op, n in sorted(op_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  op {op:<18}: {n}")

    return 1 if suspect_total else 0


# ---------------------------------------------------------------------------
# Mapping application
# ---------------------------------------------------------------------------


def load_mapping() -> list[dict]:
    if not MAPPING_PATH.exists():
        raise SystemExit(
            f"ERROR: {MAPPING_PATH} not found. Run with --generate-mapping first.")
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def apply_mapping(verbose: bool, summary: bool) -> int:
    mapping = load_mapping()

    per_file: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for e in mapping:
        per_file[REPO_ROOT / e["file"]].append((e["old"], e["new"]))

    # Conflict check 1: within a single file, no two distinct old names may
    # collapse to the same new name (would cause C++ redeclaration), unless
    # one is the LHS of an alias whose RHS is the other -> handled by alias
    # cleanup below.
    aliases_to_drop: list[tuple[Path, str, str]] = []
    fatal = 0
    for path, entries in per_file.items():
        new_to_old: dict[str, list[str]] = defaultdict(list)
        for old, new in entries:
            new_to_old[new].append(old)

        if not path.exists():
            print(f"ERROR: {path.relative_to(REPO_ROOT)} not found")
            fatal += 1
            continue

        text = path.read_text(encoding="utf-8")
        alias_pairs = {m.group("lhs"): m.group("rhs")
                       for m in ALIAS_RE.finditer(text)}

        for new, olds in new_to_old.items():
            if len(olds) <= 1:
                continue
            # Try to resolve via alias collapse.
            distinct = set(olds)
            resolved = False
            for cand in list(distinct):
                if cand in alias_pairs and alias_pairs[cand] in distinct:
                    aliases_to_drop.append((path, cand, alias_pairs[cand]))
                    distinct.discard(cand)
            if len(distinct) > 1:
                print(f"CONFLICT in {path.relative_to(REPO_ROOT)}: "
                      f"{', '.join(sorted(distinct))} all map to {new}")
                fatal += 1
            else:
                resolved = True
            if verbose and resolved:
                print(f"        alias collapse: {new} (drops {len(olds) - 1} alias(es))")

    if fatal:
        return 2

    # Conflict check 2: no symbol still references an old name that has no
    # mapping in its own file (cross-file references are not expected inside
    # archive/, but we verify).
    all_olds: dict[Path, set[str]] = {p: {o for o, _ in v} for p, v in per_file.items()}
    for path in candidate_files():
        if path in NON_KERNEL_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        local = all_olds.get(path, set())
        for other_path, names in all_olds.items():
            if other_path == path:
                continue
            for name in names:
                if name in local:
                    continue
                if re.search(rf"\b{re.escape(name)}\b", text):
                    print(f"WARNING: cross-file reference to {name} from "
                          f"{path.relative_to(REPO_ROOT)} "
                          f"(declared in {other_path.relative_to(REPO_ROOT)})")

    # Apply renames per file. Sort by descending old-name length so a short
    # name never accidentally matches inside a longer one before its turn.
    total_subs = 0
    files_changed = 0
    for path, entries in per_file.items():
        text = path.read_text(encoding="utf-8")
        original = text
        per_file_subs = 0
        for old, new in sorted(entries, key=lambda e: -len(e[0])):
            if old == new:
                continue
            pattern = re.compile(rf"\b{re.escape(old)}\b")
            text, count = pattern.subn(new, text)
            per_file_subs += count

        # Drop self-referential aliases that survive after renaming.
        if any(p == path for p, _, _ in aliases_to_drop):
            text = re.sub(
                r"^[ \t]*inline\s+constexpr\s+auto&\s+(\w+)\s*=\s*\1\s*;\s*\n?",
                "",
                text,
                flags=re.MULTILINE,
            )

        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            files_changed += 1
        total_subs += per_file_subs

        if verbose:
            tag = "OK" if text != original else "no change"
            print(f"{tag:>5}  {path.relative_to(REPO_ROOT)}: "
                  f"{per_file_subs} substitution(s)")

    # Update verify_compile.cpp sentinel symbols too.
    if VERIFY_CPP_PATH.exists():
        text = VERIFY_CPP_PATH.read_text(encoding="utf-8")
        original = text
        flat = sorted(((e["old"], e["new"]) for e in mapping
                       if e["old"] != e["new"]),
                      key=lambda e: -len(e[0]))
        v_subs = 0
        for old, new in flat:
            text, count = re.subn(rf"\b{re.escape(old)}\b", new, text)
            v_subs += count
        if text != original:
            VERIFY_CPP_PATH.write_text(text, encoding="utf-8", newline="\n")
        if verbose:
            print(f"OK    {VERIFY_CPP_PATH.relative_to(REPO_ROOT)}: "
                  f"{v_subs} substitution(s)")

    APPLIED_PATH.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(f"\nApplied: {total_subs} substitution(s) in {files_changed} file(s)")
    if aliases_to_drop:
        print(f"  removed {len(aliases_to_drop)} self-referential alias(es)")

    if summary or verbose:
        op_counter: Counter[str] = Counter()
        for e in mapping:
            op_counter[e.get("op", "unknown")] += 1
        print("\n=== summary ===")
        print(f"  symbols renamed  : {sum(1 for e in mapping if e['old'] != e['new'])}")
        print(f"  symbols unchanged: {sum(1 for e in mapping if e['old'] == e['new'])}")
        print(f"  files modified   : {files_changed}")
        for op, n in sorted(op_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  op {op:<18}: {n}")

    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate-mapping",
        action="store_true",
        help="scan archive/ and write rename_mapping.json (no files modified).",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print per-symbol details.")
    parser.add_argument("--summary", action="store_true",
                        help="print aggregate counts after the run.")
    args = parser.parse_args(argv)

    if args.generate_mapping:
        return generate_mapping(args.verbose, args.summary)
    return apply_mapping(args.verbose, args.summary)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
