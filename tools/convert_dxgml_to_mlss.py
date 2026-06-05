#!/usr/bin/env python3
"""
Convert dxGml precompiled shader headers (const unsigned char Name[N] = {...})
to MLSS shadersBin.hpp format (inline constexpr auto name = std::to_array<std::uint8_t>({...})).

Usage:
  python convert_dxgml_to_mlss.py <input.h> <output_shadersBin.hpp> <mlss_namespace>
"""

import re
import sys
import os
import json
import argparse


def extract_arrays(text):
    """
    Parse C array definitions of the form:
      const unsigned char VarName[Size] = { 0x.., ... };
    Returns list of (var_name, size, bytes_list) tuples.
    """
    results = []
    # Find all array definitions (may span multiple lines)
    pattern = re.compile(
        r'const\s+unsigned\s+char\s+(\w+)\s*\[(\d+)\]\s*=\s*\{([^}]*)\}\s*;',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        var_name = m.group(1)
        size = int(m.group(2))
        data_str = m.group(3)
        # Parse hex bytes
        hex_vals = re.findall(r'0x[0-9a-fA-F]+', data_str)
        byte_vals = [int(h, 16) for h in hex_vals]
        if len(byte_vals) != size:
            print(f"  WARNING: {var_name}: declared size {size}, got {len(byte_vals)} bytes", file=sys.stderr)
        results.append((var_name, size, byte_vals))
    return results


def bytes_to_mlss_array(var_name, byte_vals, bytes_per_line=16):
    """Format byte list as MLSS std::to_array<std::uint8_t> declaration."""
    lines = []
    lines.append(f'inline constexpr auto {var_name} = std::to_array<std::uint8_t>({{')
    for i in range(0, len(byte_vals), bytes_per_line):
        chunk = byte_vals[i:i + bytes_per_line]
        hex_str = ','.join(f'0x{b:02x}' for b in chunk)
        comma = '' if (i + bytes_per_line) >= len(byte_vals) else ','
        lines.append(f'    {hex_str}{comma}')
    lines.append('});')
    return '\n'.join(lines)


def convert_file(input_path, output_path, mlss_namespace, include_guard=None, extra_includes=None):
    """Convert a single dxGml binary header to MLSS shadersBin.hpp."""
    print(f"Converting {input_path} -> {output_path}")
    with open(input_path, 'r') as f:
        text = f.read()

    arrays = extract_arrays(text)
    if not arrays:
        print(f"  WARNING: no arrays found in {input_path}", file=sys.stderr)
        return False

    print(f"  Found {len(arrays)} arrays: {[a[0] for a in arrays]}")

    # Build output
    out_lines = []
    out_lines.append('/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved. */')
    out_lines.append('// Auto-generated from dxGml precompiled shader binaries.')
    out_lines.append('#pragma once')
    out_lines.append('')
    out_lines.append('#include <array>')
    out_lines.append('#include <cstdint>')
    if extra_includes:
        for inc in extra_includes:
            out_lines.append(f'#include {inc}')
    out_lines.append('')
    out_lines.append(f'namespace {mlss_namespace}')
    out_lines.append('{')
    out_lines.append('')

    for (var_name, size, byte_vals) in arrays:
        out_lines.append(bytes_to_mlss_array(var_name, byte_vals))
        out_lines.append('')

    out_lines.append(f'}} // namespace {mlss_namespace}')
    out_lines.append('')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(out_lines))

    print(f"  Written {output_path}")
    return True


def main():
    p = argparse.ArgumentParser(description='Convert dxGml shader headers to MLSS format')
    p.add_argument('input', help='Input dxGml header file')
    p.add_argument('output', help='Output MLSS shadersBin.hpp path')
    p.add_argument('namespace', help='MLSS C++ namespace (e.g. mlss::rmsnorm::hip::fp16::gfx1200)')
    args = p.parse_args()

    ok = convert_file(args.input, args.output, args.namespace)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
