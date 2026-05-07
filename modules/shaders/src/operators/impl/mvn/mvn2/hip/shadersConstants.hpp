/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <array>
#include <cstdint>

#include "core/core.hpp"

namespace mlss::mvn::mvn2::hip
{

    // Mirrors dxcp::HipMvn2Shader. Only the no-broadcast InstaNorm trio
    // is currently shipped; ShaderMvn2BatchNorm is a forward-looking
    // placeholder kept to stay in sync with the dxcp enum.
    enum class HipMvn2Shader : std::uint32_t
    {
        ShaderMvn2InstaNormNoBroadCastKernel1 = 0u,
        ShaderMvn2InstaNormNoBroadCastKernel2,
        ShaderMvn2InstaNormNoBroadCastKernel3,
        ShaderMvn2BatchNorm,

        ShaderCount
    };

    // Per-pipeline tile geometry, laid out for direct copy into
    // Binaries::Blob::m_constants (which is std::vector<std::uint32_t>).
    //
    // Indices follow dxcp::HipMvn2_CONSTANTS layout (HipMvn2Constants):
    //   [0x00] elementBytes
    //   [0x01] kernel1.macroTileSize  [0x02] kernel1.threadX  [0x03] kernel1.threadY  [0x04] kernel1.threadZ
    //   [0x05] kernel2.macroTileSize  [0x06] kernel2.threadX  [0x07] kernel2.threadY  [0x08] kernel2.threadZ
    //   [0x09] kernel3.macroTileSize  [0x0A] kernel3.threadX  [0x0B] kernel3.threadY  [0x0C] kernel3.threadZ

    // Number of elements processed per thread by kernel#3 along the HxW
    // axis. dxcp dispatches kernel#3 with `hw / (threadX * 4)` blocks so
    // the divisor depends on this fan-out (32 threads x 4 elements
    // = 128 elements per block on the shipped binaries).
    inline constexpr std::uint32_t kMvn2Kernel3ElementsPerThread = 4u;

    // Spatial dimension (HxW) must be a multiple of this value, matching
    // kernel1.macroTileSize.
    inline constexpr std::uint32_t kMvn2SpatialAlignment = 256u;

    // Mirrors dxcp::HipMvn2::HipMvn2_CONSTANTS verbatim. Note that
    // dxcp seeds elementBytes from sizeof(uint16) regardless of the
    // actual data type; we keep the exact same value for byte parity
    // with the reference table.
    inline constexpr std::array<std::uint32_t, 13u> HipMvn2_CONSTANTS = {
        2u,
        256u, 64u, 1u, 1u,  // kernel#1: wave64
        256u, 32u, 1u, 1u,  // kernel#2: wave32
        256u, 32u, 1u, 1u   // kernel#3: wave32, dispatch divisor = threadX * 4 = 128
    };

    // Argument layout for InstaNorm split kernel#1 (mirrors
    // dxcp::HipMvn2::Mvn2Kernel1Args). Computes per-channel partial
    // sum and partial squared sum.
    inline const std::array<MLSSarg, 4u> mvn2_instaNorm_kernel1_ARGS_CONSTANTS = {{
        {0, MLSS_FLOAT16, true,  2, true,  true,  false, "inAddr"        },
        {1, MLSS_FLOAT32, true,  2, false, false, true,  "partialSumAddr"},
        {2, MLSS_UINT32,  false, 0, true,  true,  false, "offset"        },
        {3, MLSS_UINT32,  false, 0, true,  true,  false, "hw"            }
    }};

    // Argument layout for InstaNorm split kernel#2 (mirrors
    // dxcp::HipMvn2::Mvn2Kernel2Args). Reduces partial sums into
    // mean/std per channel.
    inline const std::array<MLSSarg, 7u> mvn2_instaNorm_kernel2_ARGS_CONSTANTS = {{
        {0, MLSS_FLOAT32, true,  2, true,  true,  false, "partialSumAddr"   },
        {1, MLSS_FLOAT32, true,  2, false, false, true,  "meanStdAddr"      },
        {2, MLSS_UINT32,  false, 0, true,  true,  false, "offset"           },
        {3, MLSS_UINT32,  false, 0, true,  true,  false, "partialSumElemCnt"},
        {4, MLSS_UINT32,  false, 0, true,  true,  false, "hw"               },
        {5, MLSS_UINT32,  false, 0, true,  true,  false, "C"                },
        {6, MLSS_FLOAT32, false, 0, true,  true,  false, "eps"              }
    }};

    // Argument layout for InstaNorm split kernel#3 (mirrors
    // dxcp::HipMvn2::Mvn2Kernel3Args). Applies the normalization
    // and the (scale, bias) affine to produce the final output.
    inline const std::array<MLSSarg, 7u> mvn2_instaNorm_kernel3_ARGS_CONSTANTS = {{
        {0, MLSS_FLOAT16, true,  2, true,  true,  false, "inAddr"     },
        {1, MLSS_FLOAT16, true,  2, true,  true,  false, "scaleAddr"  },
        {2, MLSS_FLOAT16, true,  2, true,  true,  false, "biasAddr"   },
        {3, MLSS_FLOAT32, true,  2, true,  true,  false, "meanStdAddr"},
        {4, MLSS_FLOAT16, true,  2, false, false, true,  "outAddr"    },
        {5, MLSS_UINT32,  false, 0, true,  true,  false, "hw"         },
        {6, MLSS_UINT32,  false, 0, true,  true,  false, "brdcstType" }
    }};

} // namespace mlss::mvn::mvn2::hip
