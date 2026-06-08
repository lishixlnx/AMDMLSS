#!/usr/bin/env python3
"""
Generate MLSS precompiled kernel headers (shadersBin.hpp / shadersBinReloc.hpp)
from HIP (.hip), HLSL (.hlsl), and pre-assembled AMDGCN (.s) kernel sources.

Mirrors the role of dxcp/tools/metacmd/gemm/generate_header.py but targets
the MLSS archive/ format (ELF/DXIL bytes embedded as std::to_array<uint8_t>).

Usage
-----
  python generate_kernel_headers.py [options]

Options
  --hip-src DIR        Directory containing .hip kernel sources
                       (default: <repo>/src/amdxcgc/kernels/hip)
  --hlsl-src DIR       Directory containing .hlsl kernel sources
                       (default: dxcp/tools/metacmd/gemm -- the reference GEMM shaders)
  --out-dir DIR        Root output directory for generated headers
                       (default: <mlss>/archive)
  --gfx TARGETS        Semicolon-separated GPU targets for HIP kernels
                       (default: gfx1100;gfx1150;gfx1200;gfx1201)
  --hipcc PATH         Path to hipcc binary (auto-detected from ROCm if omitted)
  --ssc PATH           Path to ssc.exe HLSL compiler (auto-detected from PATH)
  --dry-run            Print commands but do not execute them
  --verbose            Print extra progress messages
  --hip-only           Only process HIP / .s kernels
  --hlsl-only          Only process HLSL kernels

Output layout (mirrors MLSS archive/ conventions)
--------------------------------------------------
  HIP kernels (per GFX target):
    archive/<op>/hip/<gfxNNNN>/fp16/shadersBin.hpp       (ET_EXEC)
    archive/<op>/hip/<gfxNNNN>/fp16/shadersBinReloc.hpp  (ET_REL)

  HLSL/DXIL kernels (single binary, DX12 runtime re-compiles per device):
    archive/gemm/hlsl/<variant_name>/shadersBin.hpp

Namespace convention
--------------------
  HIP:   archive::<op>::hip::<gfxNNNN>::fp16
  HLSL:  archive::gemm::hlsl

HLSL kernel specs (from dxcp/tools/metacmd/gemm)
-------------------------------------------------
  All 16 variants from generate_header.py are declared in HLSL_GEMM_SPECS below.
  ssc.exe compiles them to DXIL; the binary is hardware-agnostic (DX12 JITs at runtime).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults -- resolve relative to this script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
MLSS_ROOT    = SCRIPT_DIR.parent
# Sibling repos (adjust if your checkout layout differs)
REPO_ROOT    = MLSS_ROOT.parent / "AMDMIGraphX-Private"
DXCP_ROOT    = MLSS_ROOT.parent.parent / "dxcp"  # C:/Users/hisha/Documents/GitHub/dxcp

DEFAULT_HIP_SRC  = REPO_ROOT / "src" / "amdxcgc" / "kernels" / "hip"
DEFAULT_HLSL_SRC = DXCP_ROOT / "tools" / "metacmd" / "gemm"
DEFAULT_OUT_DIR  = MLSS_ROOT / "archive"
DEFAULT_GFX      = ["gfx1100", "gfx1150", "gfx1200", "gfx1201"]

# ---------------------------------------------------------------------------
# HIP kernel -> MLSS op mapping
# Maps .hip filename stem -> (amdgpu_op_name, backend)
#
# Naming convention from dxgml-dev/AMDGPUOps/amdgpuops.md:
#   Symbol: <operation>_<dtype>[_<variant>]_<arch>
#   Op:     matches amdgpu.<operation> in PDL patterns
# ---------------------------------------------------------------------------
HIP_KERNEL_MAP = {
    "gqa_kernel":                ("gqa",              "hip"),
    "rope_kernel":               ("rope",             "hip"),
    "linear_attention_kernel":   ("linear_attention", "hip"),
    "qmoe_kernel":               ("qmoe",             "hip"),
}

# Map .s file stem prefix -> amdgpu op name (for dxcp HIP_FP16 kernels)
# e.g. "gemm2d128x128x16_NN" -> op="gemm_add", variant="128x128x16_nn"
def _s_stem_to_op_variant(stem: str) -> tuple[str, str]:
    """
    Parse a dxcp .s assembly kernel filename stem into (op_name, variant).

    Input examples:
      gemm2d128x128x16_NN           -> ("gemm_add", "128x128x16_nn")
      gemm2d_NN_16x128x16_F16_gfx12 -> ("gemm_add", "16x128x16_f16_gfx12")
      gemm2d_NT_64x128x32_2_4_rocwmma_F16 -> ("gemm_add", "64x128x32_2_4_rocwmma_f16")

    The "2d" suffix is dropped; "gemm2d*" always maps to "gemm_add".
    Variant is the remainder after stripping the "gemm2d" prefix, lowercased.
    """
    s = stem.lower()
    if s.startswith("gemm2d"):
        # Strip "gemm2d" prefix, clean up separators
        variant = s[len("gemm2d"):].lstrip("_")
        return "gemm_add", variant
    # Fallback: first token is op, rest is variant
    parts = s.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

# ---------------------------------------------------------------------------
# HLSL kernel specs (ported from dxcp/tools/metacmd/gemm/generate_header.py)
# These target the DX12 GEMM metacommand (m=1 variants).
# DXIL is hardware-agnostic; DX12 re-compiles to ISA at runtime.
# ---------------------------------------------------------------------------
# Symbol naming convention per dxgml-dev/AMDGPUOps/amdgpuops.md:
#   <operation>_<dtype>_<variant>
#   operation = amdgpu op name (gemm_add for all GEMM metacommand variants)
#   dtype     = fp32 | fp16 | w4a16 | w8a16
#   variant   = layout and strategy descriptor (1xnc_tiled, 1xnr_onerow, etc.)
HLSL_GEMM_SPECS = [
    {
        "name":    "gemm_add_fp32_1xnc_tiled",
        "file":    "1xNC_Tiled.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": [],
        "comment": "amdgpu.gemm_add fp32: 1xN column-major tiled shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnc_tiled",
        "file":    "1xNC_Tiled.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN column-major tiled shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp32_1xnc_manycols",
        "file":    "1xNC_ManyCols.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": [],
        "comment": "amdgpu.gemm_add fp32: 1xN column-major many-columns shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnc_manycols",
        "file":    "1xNC_ManyCols.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN column-major many-columns shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp32_1xnc_onecol",
        "file":    "1xNC_OneCol.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": [],
        "comment": "amdgpu.gemm_add fp32: 1xN column-major one-column shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnc_onecol",
        "file":    "1xNC_OneCol.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN column-major one-column shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp32_1xnc_grouped",
        "file":    "1xNC_grouped.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp32: 1xN column-major grouped shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnc_grouped",
        "file":    "1xNC_grouped.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN column-major grouped shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp32_1xnr_tiled",
        "file":    "1xNR_Tiled.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": [],
        "comment": "amdgpu.gemm_add fp32: 1xN row-major tiled shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnr_tiled",
        "file":    "1xNR_Tiled.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN row-major tiled shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp32_1xnr_manyrows",
        "file":    "1xNR_ManyRows.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": [],
        "comment": "amdgpu.gemm_add fp32: 1xN row-major many-rows shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnr_manyrows",
        "file":    "1xNR_ManyRows.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN row-major many-rows shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp32_1xnr_onerow",
        "file":    "1xNR_OneRow.hlsl",
        "profile": "cs_6_6",
        "defines": [],
        "options": [],
        "comment": "amdgpu.gemm_add fp32: 1xN row-major one-row shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_fp16_1xnr_onerow",
        "file":    "1xNR_OneRow.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add fp16: 1xN row-major one-row shader (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_w4a16_1xnc_onecol_perchannel",
        "file":    "1xNC_OneCol_w4a16_perchannel.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add w4a16.chan: 1xN column-major int4-weight per-channel (DX12 DXIL)",
    },
    {
        "name":    "gemm_add_w8a16_1xnc_onecol_pertensor",
        "file":    "1xNC_OneCol_w8a16_pertensor.hlsl",
        "profile": "cs_6_6",
        "defines": ["DEF_FP16=1"],
        "options": ["/enable-16bit-types"],
        "comment": "amdgpu.gemm_add w8a16.tensor: 1xN column-major int8-weight per-tensor (DX12 DXIL)",
    },
]

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def find_hipcc():
    candidates = [
        "/opt/rocm/bin/hipcc",
        "C:/opt/rocm/bin/hipcc",
        shutil.which("hipcc"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def find_ssc():
    candidates = [
        shutil.which("ssc.exe"),
        shutil.which("ssc"),
        # Common dxcp build output location
        str(DXCP_ROOT / "build_msvc64" / "tools" / "ssc" / "Release" / "ssc.exe"),
        str(DXCP_ROOT / "build_msvc64" / "tools" / "ssc" / "Debug" / "ssc.exe"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def bytes_to_hpp(symbol_name: str, data: bytes, bytes_per_line: int = 16) -> str:
    lines = [f"inline constexpr auto {symbol_name} = std::to_array<std::uint8_t>({{"]
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        hex_str = ",".join(f"0x{b:02x}" for b in chunk)
        comma = "" if (i + bytes_per_line) >= len(data) else ","
        lines.append(f"    {hex_str}{comma}")
    lines.append("});")
    return "\n".join(lines)


def write_shaders_hpp(
    out_path: Path,
    namespace: str,
    entries: list[tuple[str, str, bytes]],  # (symbol, comment, data)
):
    """Write a shadersBin.hpp with one or more symbol entries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved. */",
        "// Auto-generated by tools/generate_kernel_headers.py -- do not edit.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        f"namespace {namespace}",
        "{",
        "",
    ]
    for sym, comment, data in entries:
        if comment:
            lines.append(f"// {comment}")
        lines.append(bytes_to_hpp(sym, data))
        lines.append("")
    lines.append(f"}} // namespace {namespace}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_cmd(
    cmd: list[str],
    verbose: bool,
    dry_run: bool,
    label: str = "",
) -> subprocess.CompletedProcess[str] | None:
    if verbose or dry_run:
        print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR ({label}):\n{result.stderr}", file=sys.stderr)
        return None
    return result

# ---------------------------------------------------------------------------
# HIP compilation
# ---------------------------------------------------------------------------

def compile_hip_kernel(
    src: Path,
    gfx: str,
    hipcc: str,
    relocatable: bool,
    tmpdir: Path,
    verbose: bool,
    dry_run: bool,
) -> bytes | None:
    suffix = "_reloc" if relocatable else ""
    out_path = tmpdir / f"{src.stem}_{gfx}{suffix}.hsaco"

    cmd = [hipcc, "--genco", f"--offload-arch={gfx}", "-O2", str(src), "-o", str(out_path)]
    if relocatable:
        # -fgpu-rdc produces ET_REL device code objects (needed for DX12/HipDNN reloc path)
        cmd = [hipcc, "--genco", f"--offload-arch={gfx}", "-fgpu-rdc", "-O2",
               str(src), "-o", str(out_path)]

    label = "hipcc-reloc" if relocatable else "hipcc"
    if verbose or dry_run:
        print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return b"\x7fELFdryrun"

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: hipcc failed for {src.name} ({gfx}):\n{r.stderr}", file=sys.stderr)
        return None
    return out_path.read_bytes()


# ---------------------------------------------------------------------------
# AMDGCN .s assembly -> ELF
# ---------------------------------------------------------------------------

def assemble_s_kernel(
    src: Path,
    gfx: str,
    tmpdir: Path,
    verbose: bool,
    dry_run: bool,
) -> bytes | None:
    clang = (shutil.which("clang") or shutil.which("clang-17") or
             shutil.which("clang-18") or "/opt/rocm/llvm/bin/clang")
    if not clang or not Path(clang).exists():
        print(f"  ERROR: clang not found, cannot assemble {src.name}", file=sys.stderr)
        return None

    out_path = tmpdir / f"{src.stem}_{gfx}.o"
    cmd = [clang, "-x", "assembler", "--target=amdgcn-amd-amdhsa",
           f"-mcpu={gfx}", "-c", str(src), "-o", str(out_path)]

    if verbose or dry_run:
        print(f"  [clang-asm] {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return b"\x7fELFdryrun"

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: clang asm failed for {src.name} ({gfx}):\n{r.stderr}", file=sys.stderr)
        return None
    return out_path.read_bytes()


# ---------------------------------------------------------------------------
# HLSL compilation via ssc.exe -> DXIL
# ---------------------------------------------------------------------------

def compile_hlsl(
    spec: dict,
    hlsl_src_dir: Path,
    ssc: str,
    tmpdir: Path,
    verbose: bool,
    dry_run: bool,
) -> bytes | None:
    src = hlsl_src_dir / spec["file"]
    if not src.exists() and not dry_run:
        print(f"  ERROR: HLSL source not found: {src}", file=sys.stderr)
        return None

    out_path = tmpdir / f"{spec['name']}.dxil"
    profile = spec.get("profile", "cs_6_6")

    # ssc.exe invocation mirrors generate_helper.py write_il()
    cmd = [
        ssc,
        "/input", str(src),
        "/profile", profile,
        "/entry", "main",
        "/baselogicalid", "0",
        "/dumpbin",
    ]
    if spec.get("options"):
        cmd += ["/options", " ".join(spec["options"])]
    if spec.get("defines"):
        # ssc.exe takes /define as a single space-separated string
        cmd += ["/define", " ".join(spec["defines"])]

    if verbose or dry_run:
        print(f"  [ssc] {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return b"\x44\x58\x42\x43dryrun"  # DXBC magic

    # ssc writes dxil_<profile>_main.h in CWD; run in tmpdir
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmpdir))
    if r.returncode != 0:
        print(f"  ERROR: ssc failed for {spec['file']}:\n{r.stderr}", file=sys.stderr)
        return None

    # ssc dumps DXIL as a C array in dxil_<profile>_main.h
    dxil_h = tmpdir / f"dxil_{profile}_main.h"
    if not dxil_h.exists():
        print(f"  ERROR: ssc did not produce {dxil_h.name}", file=sys.stderr)
        return None

    # Parse the C array from ssc output into raw bytes
    text = dxil_h.read_text(encoding="utf-8")
    hex_vals = re.findall(r"0x[0-9a-fA-F]+", text)
    raw = bytes(int(h, 16) for h in hex_vals)

    # Clean up ssc temp files
    for stem in (f"dxil_{profile}_main.h", f"amdil_{profile}_main.h", f"{profile}_main.txt"):
        tmp_f = tmpdir / stem
        if tmp_f.exists():
            tmp_f.unlink(missing_ok=True)

    return raw


# ---------------------------------------------------------------------------
# Main processing functions
# ---------------------------------------------------------------------------

def process_hip_kernels(args, tmpdir: Path):
    hip_src_dir = Path(args.hip_src)
    if not hip_src_dir.exists():
        print(f"HIP source directory not found: {hip_src_dir}", file=sys.stderr)
        return

    hipcc = args.hipcc or find_hipcc()
    if not hipcc and not args.dry_run:
        print("ERROR: hipcc not found. Install ROCm or pass --hipcc <path>.", file=sys.stderr)
        return
    hipcc = hipcc or "<hipcc>"

    gfx_targets = [t.strip() for t in args.gfx.split(";") if t.strip()]

    for hip_file in sorted(hip_src_dir.glob("*.hip")):
        stem = hip_file.stem
        mapping = HIP_KERNEL_MAP.get(stem)
        if not mapping:
            if args.verbose:
                print(f"  SKIP: {hip_file.name} (not in HIP_KERNEL_MAP)")
            continue

        op, backend = mapping
        print(f"\n[HIP] {hip_file.name} -> archive/{op}/{backend}/")

        for gfx in gfx_targets:
            for reloc in (False, True):
                kind  = "reloc" if reloc else "shared"
                print(f"  Compiling {gfx} ({kind})...")

                data = compile_hip_kernel(hip_file, gfx, hipcc, reloc, tmpdir,
                                          args.verbose, args.dry_run)
                if data is None:
                    continue

                # Naming per amdgpuops.md: <op>_<dtype>[_reloc]_<arch>
                reloc_tag = "_reloc" if reloc else ""
                symbol    = f"{op}_fp16{reloc_tag}_{gfx}"
                out_dir   = Path(args.out_dir) / op / backend / gfx / "fp16"
                hpp_name  = "shadersBinReloc.hpp" if reloc else "shadersBin.hpp"
                out_path  = out_dir / hpp_name
                namespace = f"archive::{op}::{backend}::{gfx}::fp16"

                if not args.dry_run:
                    write_shaders_hpp(out_path, namespace, [(symbol, "", data)])
                    print(f"  -> {out_path.relative_to(MLSS_ROOT)}")
                else:
                    print(f"  -> (dry-run) {out_dir.relative_to(MLSS_ROOT)}/{hpp_name}")


def _infer_s_op_backend_dtype(s_file: Path, gfx_targets: list[str]) -> tuple[str, str, str]:
    """
    Infer (op, backend, dtype) from a .s file's parent directory path.

    Handles two layouts:
      dxcp HIP_FP16/gfx1201/[activations|no-activations]/<file>.s  -> gemm/hip/fp16
      dxcp CK_FP32/gfx1201/<file>.s                                  -> gemm/ck/fp32
      amdxcgc hip_kernels/<file>.s                                    -> <stem>/hip/fp16
    """
    parts = [p.lower() for p in s_file.parts]

    # Check for dxcp HIP_FP16 layout
    if any("hip_fp16" in p for p in parts):
        return "gemm", "hip", "fp16"

    # Check for dxcp CK_FP32 layout
    if any("ck_fp32" in p or ("ck" in p and "fp32" in p) for p in parts):
        return "gemm", "ck", "fp32"

    # Fallback: derive from immediate parent name
    parent = s_file.parent.name.lower()
    dtype   = "fp32" if "fp32" in parent or "fp32" in s_file.stem.lower() else "fp16"
    backend = "ck"  if "ck" in parent else "hip"
    op      = parent if "fp" not in parent else s_file.parent.parent.name.lower()
    return op, backend, dtype


def process_s_kernels(args, tmpdir: Path):
    """
    Assemble pre-written AMDGCN .s files found recursively under args.hip_src.

    All .s files that share the same output path (op/backend/gfx/dtype/shadersBin.hpp)
    are collected first, then written to a single header with all symbols.
    """
    search_dir = Path(args.hip_src)
    if not search_dir.exists():
        return

    gfx_targets = [t.strip() for t in args.gfx.split(";") if t.strip()]

    # Collect: out_path -> [(symbol, comment, data)]
    from collections import defaultdict
    pending: dict[Path, list[tuple[str, str, bytes]]] = defaultdict(list)
    # Also track namespace per out_path
    namespaces: dict[Path, str] = {}

    for s_file in sorted(search_dir.rglob("*.s")):
        # Infer gfx from filename if embedded (e.g. *_gfx1201.s)
        gfx_in_name = next((g for g in gfx_targets if g in s_file.stem), None)
        # Also check parent dir name for gfx (e.g. HIP_FP16/gfx1201/...)
        gfx_in_dir  = next((g for g in gfx_targets if g in s_file.parent.name), None)
        gfx_hint    = gfx_in_name or gfx_in_dir
        targets     = [gfx_hint] if gfx_hint else gfx_targets

        op, backend, dtype = _infer_s_op_backend_dtype(s_file, gfx_targets)

        print(f"\n[ASM] {s_file.name}  ({op}/{backend}/{dtype})")
        for gfx in targets:
            data = assemble_s_kernel(s_file, gfx, tmpdir, args.verbose, args.dry_run)
            if data is None:
                continue

            # Build symbol per amdgpuops.md: <op>_<dtype>_<variant>_<arch>
            # Parse the .s stem for its tile-size variant (e.g. 128x128x16_nn)
            inferred_op, tile_variant = _s_stem_to_op_variant(s_file.stem)
            if op == "gemm" or op == "gemm_add":
                op = "gemm_add"   # normalise to amdgpu op name

            # Activation hint from directory path
            act_tag = ""
            for part in s_file.parts:
                pl = part.lower()
                if "no-activation" in pl or "noact" in pl or "no_act" in pl:
                    act_tag = "_noact"
                    break
                if "activation" in pl and "no" not in pl:
                    act_tag = "_act"
                    break

            variant_part = f"_{tile_variant}" if tile_variant else ""
            # Don't double-append arch if it's already embedded in the variant
            # (e.g. gemm2d_NN_16x128x16_F16_gfx12 already carries "gfx12")
            gfx_already_in_variant = any(
                tile_variant.endswith(g) or f"_{g}" in tile_variant
                for g in gfx_targets
            )
            arch_suffix = "" if gfx_already_in_variant else f"_{gfx}"
            symbol    = f"{op}_{dtype}{variant_part}{act_tag}{arch_suffix}"
            out_dir   = Path(args.out_dir) / op / backend / gfx / dtype
            out_path  = out_dir / "shadersBin.hpp"
            namespace = f"archive::{op}::{backend}::{gfx}::{dtype}"

            namespaces[out_path] = namespace
            pending[out_path].append((symbol, "", data))

            if args.dry_run:
                print(f"  -> (dry-run) {out_dir.relative_to(MLSS_ROOT)}/shadersBin.hpp")

    if not args.dry_run:
        for out_path, entries in pending.items():
            write_shaders_hpp(out_path, namespaces[out_path], entries)
            print(f"  -> {out_path.relative_to(MLSS_ROOT)}  ({len(entries)} symbols)")


def process_hlsl_kernels(args, tmpdir: Path):
    """
    Compile all 16 HLSL GEMM variants from dxcp/tools/metacmd/gemm/
    into DXIL and embed them in MLSS archive/gemm/hlsl/shadersBin.hpp.

    DXIL is hardware-agnostic -- DX12 re-compiles to ISA at runtime for every
    gfx1100+ device, so there is only ONE output file (no per-GFX split).
    """
    hlsl_src_dir = Path(args.hlsl_src)
    if not hlsl_src_dir.exists():
        print(f"HLSL source directory not found: {hlsl_src_dir}", file=sys.stderr)
        print("  Pass --hlsl-src <path> pointing to dxcp/tools/metacmd/gemm/",
              file=sys.stderr)
        return

    ssc = args.ssc or find_ssc()
    if not ssc and not args.dry_run:
        print("WARNING: ssc.exe not found; skipping HLSL kernels.", file=sys.stderr)
        print("  Build ssc from dxcp or pass --ssc <path>.", file=sys.stderr)
        return
    ssc = ssc or "<ssc.exe>"

    entries: list[tuple[str, str, bytes]] = []
    failed: list[str] = []

    print(f"\n[HLSL] Compiling {len(HLSL_GEMM_SPECS)} GEMM variants from {hlsl_src_dir}")
    for spec in HLSL_GEMM_SPECS:
        print(f"  {spec['name']} ({spec['file']})...")
        data = compile_hlsl(spec, hlsl_src_dir, ssc, tmpdir, args.verbose, args.dry_run)
        if data is None:
            failed.append(spec["name"])
            continue
        entries.append((spec["name"], spec["comment"], data))

    if entries:
        # All 16 variants go into a single header (no per-GFX split for DXIL)
        out_path  = Path(args.out_dir) / "gemm" / "hlsl" / "shadersBin.hpp"
        namespace = "archive::gemm::hlsl"

        if not args.dry_run:
            write_shaders_hpp(out_path, namespace, entries)
            print(f"  -> {out_path.relative_to(MLSS_ROOT)}  ({len(entries)} symbols)")
        else:
            print(f"  -> (dry-run) archive/gemm/hlsl/shadersBin.hpp  ({len(entries)} symbols)")

    if failed:
        print(f"\nFailed variants: {failed}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate MLSS precompiled kernel headers from HIP/HLSL/.s sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--hip-src",  default=str(DEFAULT_HIP_SRC),
                   help=f"Directory of .hip sources (default: {DEFAULT_HIP_SRC})")
    p.add_argument("--hlsl-src", default=str(DEFAULT_HLSL_SRC),
                   help=f"Directory of .hlsl sources (default: {DEFAULT_HLSL_SRC})")
    p.add_argument("--s-src",    default=None,
                   help="Extra directory to search for pre-assembled .s files "
                        "(e.g. dxcp/tools/metacmd/gemm/HIP_FP16). "
                        "Scanned recursively; gfx target inferred from filename or --gfx list.")
    p.add_argument("--out-dir",  default=str(DEFAULT_OUT_DIR),
                   help=f"Root output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--gfx",      default=";".join(DEFAULT_GFX),
                   help="Semicolon-separated HIP GPU targets (HLSL/DXIL ignores this)")
    p.add_argument("--hipcc",    default=None, help="Path to hipcc binary")
    p.add_argument("--ssc",      default=None, help="Path to ssc.exe HLSL compiler")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print commands without executing or writing files")
    p.add_argument("--verbose",  action="store_true", help="Extra progress output")
    p.add_argument("--hip-only", action="store_true",
                   help="Only process HIP kernels (skip HLSL)")
    p.add_argument("--hlsl-only", action="store_true",
                   help="Only process HLSL kernels (skip HIP)")
    args = p.parse_args()

    if args.dry_run:
        print("=== DRY RUN -- no files will be written ===\n")

    with tempfile.TemporaryDirectory(prefix="mlss_kernels_") as tmp:
        tmpdir = Path(tmp)

        if not args.hlsl_only:
            process_hip_kernels(args, tmpdir)
            # Search --hip-src and optionally --s-src for .s files
            process_s_kernels(args, tmpdir)
            if args.s_src:
                # Temporarily redirect hip_src to s_src for the .s scan
                s_args = argparse.Namespace(**vars(args))
                s_args.hip_src = args.s_src
                process_s_kernels(s_args, tmpdir)

        if not args.hip_only:
            process_hlsl_kernels(args, tmpdir)

    print("\nDone.")
    print("\nNext steps:")
    print("  1. Review generated headers in archive/<op>/hip/<gfx>/fp16/")
    print("  2. #include them in modules/shaders/src/operators/impl/<op>/hip/shadersUtils.cpp")
    print("  3. Update the namespace aliases and decision trees in shadersUtils.cpp")
    print("  4. Rebuild AMDMLSS and run unit tests")


if __name__ == "__main__":
    main()
