#!/usr/bin/env python3
"""Convert legacy MVN2 archive shader binaries to the modern StaticShaderType format."""

import re
import sys
import os

def convert_archive(input_path, output_path, namespace):
    with open(input_path, "r") as f:
        content = f.read()

    pattern = r"const unsigned char\s+(\w+)\[(\d+)\]\s*=\s*\{([^;]+)\};"
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        print(f"ERROR: No arrays found in {input_path}")
        sys.exit(1)

    lines = [
        "/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */",
        "#pragma once",
        "",
        '#include "shaders/shaders.hpp"',
        "",
        f"namespace mlss::mvn::mvn2::hip::fp16::{namespace}",
        "{",
        "",
    ]

    for m in matches:
        name = m.group(1)
        size = m.group(2)
        data = m.group(3).strip()

        lines.append(f"const StaticShaderType<{size}> {name} =")
        lines.append("{")
        lines.append("    .m_binary = {")

        raw_bytes = [b.strip() for b in data.split(",") if b.strip()]
        for i in range(0, len(raw_bytes), 16):
            chunk = ",".join(raw_bytes[i : i + 16])
            if i + 16 < len(raw_bytes):
                chunk += ","
            lines.append(chunk)

        lines.append("    },")
        lines.append(f'    .m_kernelName = "{name}",')
        lines.append('    .m_compilerVersion = "archive",')
        lines.append("    .m_codeObjectVersion = 5,")
        lines.append("    .m_isRelocatable = true,")
        lines.append("    .m_shaderType = ShaderTypesFlags::UNKNOWN")
        lines.append("};")
        lines.append("")

    lines.append(f"}} // namespace mlss::mvn::mvn2::hip::fp16::{namespace}")
    lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"Wrote {output_path} ({len(matches)} shaders)")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)

    convert_archive(
        "archive/mvn/mvn2/gfx1100/shadersBinReloc.hpp",
        "modules/shaders/src/operators/impl/mvn/mvn2/hip/gfx1100/fp16/shadersBin.hpp",
        "gfx1100",
    )
    convert_archive(
        "archive/mvn/mvn2/gfx1150/shadersBinReloc.hpp",
        "modules/shaders/src/operators/impl/mvn/mvn2/hip/gfx1150/fp16/shadersBin.hpp",
        "gfx1150",
    )
    convert_archive(
        "archive/mvn/mvn2/gfx1200/metaCmdBinHipMvn2Gfx1200.h",
        "modules/shaders/src/operators/impl/mvn/mvn2/hip/gfx1200/fp16/shadersBin.hpp",
        "gfx1200",
    )
