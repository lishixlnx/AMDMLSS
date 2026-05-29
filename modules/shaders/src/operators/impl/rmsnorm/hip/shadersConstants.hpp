/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <array>
#include <cstdint>

#include "core/core.hpp"

namespace mlss::rmsnorm::hip
{

    // Mirrors dxcp::RmsNormShader. Each variant processes a different number
    // of rows (blockM) per thread block; the Pad variants handle the case
    // where M is not evenly divisible by blockM.
    enum class HipRmsNormShader : std::uint32_t
    {
        ShaderBm1Bn128    = 0u,  // blockM=1,  M==1
        ShaderBm2Bn128    = 1u,  // blockM=2,  1<M<16, M even
        ShaderBm2Bn128Pad = 2u,  // blockM=2,  1<M<16, M odd
        ShaderBm8Bn128    = 3u,  // blockM=8,  M>=16, M%8==0
        ShaderBm8Bn128Pad = 4u,  // blockM=8,  M>=16, M%8!=0

        ShaderCount
    };

    // Per-shader tile geometry for Binaries::Blob::m_constants.
    // Indices mirror dxcp::RmsNormConstants:
    //   [0] threadX  [1] threadY  [2] threadZ  [3] blockM  [4] blockN

    inline constexpr std::array<std::uint32_t, 5u> RmsNorm_Bm1_Bn128_CONSTANTS    = {128u,  1u, 1u, 1u, 128u};
    inline constexpr std::array<std::uint32_t, 5u> RmsNorm_Bm2_Bn128_CONSTANTS    = {256u,  1u, 1u, 2u, 128u};
    inline constexpr std::array<std::uint32_t, 5u> RmsNorm_Bm2_Bn128Pad_CONSTANTS = {256u,  1u, 1u, 2u, 128u};
    inline constexpr std::array<std::uint32_t, 5u> RmsNorm_Bm8_Bn128_CONSTANTS    = {1024u, 1u, 1u, 8u, 128u};
    inline constexpr std::array<std::uint32_t, 5u> RmsNorm_Bm8_Bn128Pad_CONSTANTS = {1024u, 1u, 1u, 8u, 128u};

    // Argument layout for every RmsNorm fp16 binary.
    // Mirrors dxcp::RmsNormArgs: xAddr, gammaAddr, yAddr, epsilon, m, n, strideX, strideY.
    inline const std::array<MLSSarg, 8u> hip_rmsnorm_ARGS_CONSTANTS = {{
        {0, MLSS_FLOAT16, true,  2, true,  true,  false, "xAddr"    },
        {1, MLSS_FLOAT16, true,  2, true,  true,  false, "gammaAddr"},
        {2, MLSS_FLOAT16, true,  2, false, false, true,  "yAddr"    },
        {3, MLSS_FLOAT32, false, 0, true,  true,  false, "epsilon"  },
        {4, MLSS_UINT32,  false, 0, true,  true,  false, "m"        },
        {5, MLSS_UINT32,  false, 0, true,  true,  false, "n"        },
        {6, MLSS_UINT32,  false, 0, true,  true,  false, "strideX"  },
        {7, MLSS_UINT32,  false, 0, true,  true,  false, "strideY"  }
    }};

} // namespace mlss::rmsnorm::hip
