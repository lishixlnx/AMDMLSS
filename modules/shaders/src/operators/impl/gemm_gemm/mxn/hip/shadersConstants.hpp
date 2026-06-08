/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <array>
#include <cstdint>

#include "core/core.hpp"

namespace mlss::gemm_gemm::mxn::hip
{

    // Mirrors dxcp::GemmGemmShader. The 12 variants span 3 tile configs × 4 transposition modes.
    // Transposition suffix key: (none)=NN, Transb=B both transposed, Transb0=only B0, Transb1=only B1.
    enum class HipGemmGemmShader : std::uint32_t
    {
        Shader16x64x64x32WMMA     = 0u,   // Wave0 Bs32 Lper32 NN
        Shader16x64x64x32TBWMMA   = 1u,   // Wave0 Bs32 Lper32 Transb  (B0+B1 transposed)
        Shader16x64x64x32TB0WMMA  = 2u,   // Wave0 Bs32 Lper32 Transb0 (B0 only)
        Shader16x64x64x32TB1WMMA  = 3u,   // Wave0 Bs32 Lper32 Transb1 (B1 only)
        Shader16x64x64x64WMMA     = 4u,   // Wave1 Bs32 Lper64 NN
        Shader16x64x64x64TBWMMA   = 5u,   // Wave1 Bs32 Lper64 Transb
        Shader16x64x64x64TB0WMMA  = 6u,   // Wave1 Bs32 Lper64 Transb0
        Shader16x64x64x64TB1WMMA  = 7u,   // Wave1 Bs32 Lper64 Transb1
        Shader32x64x64x128WMMA    = 8u,   // Wave2 Bs64 Lper128 NN
        Shader32x64x64x128TBWMMA  = 9u,   // Wave2 Bs64 Lper128 Transb
        Shader32x64x64x128TB0WMMA = 10u,  // Wave2 Bs64 Lper128 Transb0
        Shader32x64x64x128TB1WMMA = 11u,  // Wave2 Bs64 Lper128 Transb1

        ShaderCount
    };

    // Per-shader tile geometry, laid out for direct copy into
    // Binaries::Blob::m_constants (which is std::vector<std::uint32_t>).
    //
    // Indices follow dxcp::GemmGemmConstants:
    //   [0] activationElementBytes  [1] macroTileM  [2] macroTileN
    //   [3] macroTileL              [4] macroTileK
    //   [5] threadX  [6] threadY  [7] threadZ

    // Wave0 / Bs32 / LperBlock=32  (M=16, N=64, L=32, K=64, threadX=32)
    inline constexpr std::array<std::uint32_t, 8u> GemmGemm_16x64x64x32_CONSTANTS = {2u, 16u, 64u, 32u, 64u, 32u, 1u, 1u};

    // Wave1 / Bs32 / LperBlock=64  (M=16, N=64, L=64, K=64, threadX=32)
    inline constexpr std::array<std::uint32_t, 8u> GemmGemm_16x64x64x64_CONSTANTS = {2u, 16u, 64u, 64u, 64u, 32u, 1u, 1u};

    // Wave2 / Bs64 / LperBlock=128 (M=32, N=64, L=128, K=64, threadX=64)
    inline constexpr std::array<std::uint32_t, 8u> GemmGemm_32x64x64x128_CONSTANTS = {2u, 32u, 64u, 128u, 64u, 64u, 1u, 1u};

    // Argument layout for GemmGemm fp16 binaries.
    // Mirrors dxcp::GemmGemmArgs: AAddr, B0Addr, B1Addr, YAddr, M, L, K, N, batch,
    // strideA, strideB0, strideB1, strideY, batchStrideA, batchStrideB0, batchStrideB1, batchStrideY,
    // hasMainKBlockLoop, tailNumber.
    inline const std::array<MLSSarg, 19u> hip_gemm_gemm_fp16_ARGS_CONSTANTS = {{
        { 0, MLSS_FLOAT16, true,  2, true,  true,  false, "AAddr"          },
        { 1, MLSS_FLOAT16, true,  2, true,  true,  false, "B0Addr"         },
        { 2, MLSS_FLOAT16, true,  2, true,  true,  false, "B1Addr"         },
        { 3, MLSS_FLOAT16, true,  2, false, false, true,  "YAddr"          },
        { 4, MLSS_UINT32,  false, 0, true,  true,  false, "M"              },
        { 5, MLSS_UINT32,  false, 0, true,  true,  false, "L"              },
        { 6, MLSS_UINT32,  false, 0, true,  true,  false, "K"              },
        { 7, MLSS_UINT32,  false, 0, true,  true,  false, "N"              },
        { 8, MLSS_UINT32,  false, 0, true,  true,  false, "batch"          },
        { 9, MLSS_UINT32,  false, 0, true,  true,  false, "strideA"        },
        {10, MLSS_UINT32,  false, 0, true,  true,  false, "strideB0"       },
        {11, MLSS_UINT32,  false, 0, true,  true,  false, "strideB1"       },
        {12, MLSS_UINT32,  false, 0, true,  true,  false, "strideY"        },
        {13, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideA"   },
        {14, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideB0"  },
        {15, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideB1"  },
        {16, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideY"   },
        {17, MLSS_UINT32,  false, 0, true,  true,  false, "hasMainKBlockLoop"},
        {18, MLSS_INT32,   false, 0, true,  true,  false, "tailNumber"     }
    }};

    // Argument layout for GemmGemmW8A16PerTensor binaries.
    // Extends GemmGemmArgs with 4 additional dequant pointers:
    // scale0Addr, zp0Addr, scale1Addr, zp1Addr.
    inline const std::array<MLSSarg, 23u> hip_gemm_gemm_w8a16_ARGS_CONSTANTS = {{
        { 0, MLSS_FLOAT16, true,  2, true,  true,  false, "AAddr"          },
        { 1, MLSS_INT8,    true,  2, true,  true,  false, "B0Addr"         },
        { 2, MLSS_INT8,    true,  2, true,  true,  false, "B1Addr"         },
        { 3, MLSS_FLOAT16, true,  2, false, false, true,  "YAddr"          },
        { 4, MLSS_UINT32,  false, 0, true,  true,  false, "M"              },
        { 5, MLSS_UINT32,  false, 0, true,  true,  false, "L"              },
        { 6, MLSS_UINT32,  false, 0, true,  true,  false, "K"              },
        { 7, MLSS_UINT32,  false, 0, true,  true,  false, "N"              },
        { 8, MLSS_UINT32,  false, 0, true,  true,  false, "batch"          },
        { 9, MLSS_UINT32,  false, 0, true,  true,  false, "strideA"        },
        {10, MLSS_UINT32,  false, 0, true,  true,  false, "strideB0"       },
        {11, MLSS_UINT32,  false, 0, true,  true,  false, "strideB1"       },
        {12, MLSS_UINT32,  false, 0, true,  true,  false, "strideY"        },
        {13, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideA"   },
        {14, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideB0"  },
        {15, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideB1"  },
        {16, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideY"   },
        {17, MLSS_UINT32,  false, 0, true,  true,  false, "hasMainKBlockLoop"},
        {18, MLSS_INT32,   false, 0, true,  true,  false, "tailNumber"     },
        {19, MLSS_FLOAT32, true,  2, true,  true,  false, "scale0Addr"     },
        {20, MLSS_INT8,    true,  2, true,  true,  false, "zp0Addr"        },
        {21, MLSS_FLOAT32, true,  2, true,  true,  false, "scale1Addr"     },
        {22, MLSS_INT8,    true,  2, true,  true,  false, "zp1Addr"        }
    }};

} // namespace mlss::gemm_gemm::mxn::hip
