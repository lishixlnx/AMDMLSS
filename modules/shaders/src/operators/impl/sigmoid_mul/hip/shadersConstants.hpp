/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <array>
#include <cstdint>

#include "core/core.hpp"

namespace mlss::sigmoid_mul::hip
{

    // Per-kernel tile geometry for Binaries::Blob::m_constants.
    // Indices mirror dxcp::SigmoidMulConstants:
    //   [0] blockSize (threads per block in X)
    //   [1] threadY   [2] threadZ
    inline constexpr std::array<std::uint32_t, 3u> SigmoidMul_CONSTANTS = {64u, 1u, 1u};

    // Argument layout for the SigmoidMul fp16 binary.
    // Mirrors dxcp::SigmoidMulArgs: aAddr, bAddr, outAddr, n, c, h, w,
    // strideA×4 (n,c,h,w), strideB×4, strideOut×4.
    inline const std::array<MLSSarg, 19u> hip_sigmoid_mul_ARGS_CONSTANTS = {{
        { 0, MLSS_FLOAT16, true,  2, true,  true,  false, "aAddr"       },
        { 1, MLSS_FLOAT16, true,  2, true,  true,  false, "bAddr"       },
        { 2, MLSS_FLOAT16, true,  2, false, false, true,  "outAddr"     },
        { 3, MLSS_UINT32,  false, 0, true,  true,  false, "n"           },
        { 4, MLSS_UINT32,  false, 0, true,  true,  false, "c"           },
        { 5, MLSS_UINT32,  false, 0, true,  true,  false, "h"           },
        { 6, MLSS_UINT32,  false, 0, true,  true,  false, "w"           },
        { 7, MLSS_UINT32,  false, 0, true,  true,  false, "strideA_n"   },
        { 8, MLSS_UINT32,  false, 0, true,  true,  false, "strideA_c"   },
        { 9, MLSS_UINT32,  false, 0, true,  true,  false, "strideA_h"   },
        {10, MLSS_UINT32,  false, 0, true,  true,  false, "strideA_w"   },
        {11, MLSS_UINT32,  false, 0, true,  true,  false, "strideB_n"   },
        {12, MLSS_UINT32,  false, 0, true,  true,  false, "strideB_c"   },
        {13, MLSS_UINT32,  false, 0, true,  true,  false, "strideB_h"   },
        {14, MLSS_UINT32,  false, 0, true,  true,  false, "strideB_w"   },
        {15, MLSS_UINT32,  false, 0, true,  true,  false, "strideOut_n" },
        {16, MLSS_UINT32,  false, 0, true,  true,  false, "strideOut_c" },
        {17, MLSS_UINT32,  false, 0, true,  true,  false, "strideOut_h" },
        {18, MLSS_UINT32,  false, 0, true,  true,  false, "strideOut_w" }
    }};

} // namespace mlss::sigmoid_mul::hip
