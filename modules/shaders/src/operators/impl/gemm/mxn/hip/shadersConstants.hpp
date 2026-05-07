/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <array>
#include <cstdint>

#include "core/core.hpp"

namespace mlss::gemm::mxn::hip
{

    // Mirrors dxcp::HipGemmShader. Each entry is the macro tile recipe
    // applied by the corresponding gemm_add_fp16_<tile>_*_gfxNNNN binary.
    enum class HipGemmShader : std::uint32_t
    {
        Shader16x16x16WMMA_NN = 0u,
        Shader16x128x16WMMA_NN,
        Shader16x256x16WMMA_NN,
        Shader64x128x32WMMA_NN,
        Shader128x128x16WMMA_NN,
        Shader128x64x32WMMA_NN,
        Shader256x16x16WMMA_NN,
        Shader64x128x32WMMA_NT,

        ShaderCount
    };

    // Per-shader tile geometry, laid out for direct copy into
    // Binaries::Blob::m_constants (which is std::vector<std::uint32_t>).
    // Indices follow dxcp::HipGemm::gemm2d_*_F16_F16_CONSTANTS:
    //   [0] elementBytes  [1] macroTileM  [2] macroTileN  [3] macroTileK
    //   [4] threadX       [5] threadY     [6] threadZ

    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_16x16x16NN   = {2u,  16u,  16u, 16u,  32u, 1u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_16x128x16NN  = {2u,  16u, 128u, 16u,  32u, 4u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_16x256x16NN  = {2u,  16u, 256u, 16u,  32u, 8u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_64x128x32NN  = {2u,  64u, 128u, 32u,  64u, 2u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_128x128x16NN = {2u, 128u, 128u, 16u, 128u, 2u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_128x64x32NN  = {2u, 128u,  64u, 32u,  64u, 2u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_256x16x16NN  = {2u, 256u,  16u, 16u, 256u, 1u, 1u};
    inline constexpr std::array<std::uint32_t, 7u> Gemm2d_64x128x32NT  = {2u,  64u, 128u, 32u,  64u, 2u, 1u};

    // Argument layout for every gemm_add_fp16_<tile>_*_gfxNNNN binary
    // (mirrors dxcp::HipGemm::GemmArgs and matches the conv1x1 WMMA
    // arg list, since both share the same kernel signature).
    inline const std::array<MLSSarg, 17u> hip_gemm_mxn_ARGS_CONSTANTS = {{
        { 0, MLSS_UINT32,  false, 0, true,  true,  false, "M"            },
        { 1, MLSS_UINT32,  false, 0, true,  true,  false, "N"            },
        { 2, MLSS_UINT32,  false, 0, true,  true,  false, "K"            },
        { 3, MLSS_FLOAT16, true,  2, true,  true,  false, "AAddr"        },
        { 4, MLSS_FLOAT16, true,  2, true,  true,  false, "BAddr"        },
        { 5, MLSS_FLOAT16, true,  2, true,  true,  false, "CAddr"        },
        { 6, MLSS_FLOAT16, true,  2, false, false, true,  "DAddr"        },
        { 7, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideA" },
        { 8, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideB" },
        { 9, MLSS_UINT32,  false, 0, true,  true,  false, "batchStrideC" },
        {10, MLSS_FLOAT32, false, 0, true,  true,  false, "alpha"        },
        {11, MLSS_FLOAT32, false, 0, true,  true,  false, "beta"         },
        {12, MLSS_UINT32,  false, 0, true,  true,  false, "activation"   },
        {13, MLSS_FLOAT32, false, 0, true,  true,  false, "param1"       },
        {14, MLSS_FLOAT32, false, 0, true,  true,  false, "param2"       },
        {15, MLSS_UINT32,  false, 0, true,  true,  false, "broadcastN"   },
        {16, MLSS_UINT32,  false, 0, true,  true,  false, "broadcastM"   }
    }};

} // namespace mlss::gemm::mxn::hip
