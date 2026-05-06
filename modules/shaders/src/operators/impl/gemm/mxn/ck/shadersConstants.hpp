/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <array>
#include <cstdint>

#include "core/core.hpp"

namespace mlss::gemm::mxn::ck
{

    // Mirrors dxcp::CKGemmShader. Predicting class 0 means "skip our path
    // and let the MSFT shader run". We only ship the FP32_NN binary.
    enum class CKGemmShader : std::int32_t
    {
        ShaderSkipMetaCmd = 0,
        ShaderFp32_NN     = 1,
        ShaderCount
    };

    // Per-shader tile geometry, laid out for direct copy into
    // Binaries::Blob::m_constants (which is std::vector<std::uint32_t>).
    // Indices follow dxcp::CKGemm::gemm2d_FP32_CONSTANTS:
    //   [0] elementBytes  [1] macroTileM  [2] macroTileN  [3] macroTileK
    //   [4] threadX       [5] threadY     [6] threadZ

    // Mirrors dxcp::CKGemm::gemm2d_FP32_CONSTANTS verbatim. Note that
    // dxcp seeds elementBytes from sizeof(uint16) even on the FP32 path;
    // we keep the exact same value to stay byte-compatible with the
    // reference table.
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_FP32 = {
        2u, 128u, 128u, 1u, 256u, 1u, 1u
    };

    // Argument layout for the gemm_add_fp32_add_dl_nn_gfx1201 binary
    // (mirrors dxcp::CKGemm::CKGemmArgs / DdiMetaCmdGemmCKMxN dispatch).
    inline const std::array<MLSSarg, 9u> ck_gemm_mxn_ARGS_CONSTANTS = {{
        {0, MLSS_FLOAT32, true,  2, true,  true,  false, "AAddr"  },
        {1, MLSS_FLOAT32, true,  2, true,  true,  false, "BAddr"  },
        {2, MLSS_FLOAT32, true,  2, false, false, true,  "CAddr"  },
        {3, MLSS_UINT32,  false, 0, true,  true,  false, "M"      },
        {4, MLSS_UINT32,  false, 0, true,  true,  false, "N"      },
        {5, MLSS_UINT32,  false, 0, true,  true,  false, "K"      },
        {6, MLSS_UINT32,  false, 0, true,  true,  false, "strideA"},
        {7, MLSS_UINT32,  false, 0, true,  true,  false, "strideB"},
        {8, MLSS_UINT32,  false, 0, true,  true,  false, "strideC"}
    }};

} // namespace mlss::gemm::mxn::ck
