/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::conv::dilated::hip::wmma::fp16
{

    constexpr auto gemm2d_32x32x64NN_F16_F16_CONSTANTS   = std::to_array<std::uint32_t>({ sizeof(std::uint16_t), 32, 32, 64, 64, 1, 1 });
    constexpr auto gemm2d_64x32x32NN_F16_F16_CONSTANTS   = std::to_array<std::uint32_t>({ sizeof(std::uint16_t), 64, 32, 32, 64, 1, 1 });
    constexpr auto gemm2d_64x64x64NN_F16_F16_CONSTANTS   = std::to_array<std::uint32_t>({ sizeof(std::uint16_t), 64, 64, 64, 128, 1, 1 });
    constexpr auto gemm2d_128x64x32NN_F16_F16_CONSTANTS  = std::to_array<std::uint32_t>({ sizeof(std::uint16_t), 128, 64, 32, 128, 1, 1 });
    constexpr auto gemm2d_256x32x32NN_F16_F16_CONSTANTS  = std::to_array<std::uint32_t>({ sizeof(std::uint16_t), 256, 32, 32, 128, 1, 1 });
    constexpr auto gemm2d_128x128x32NN_F16_F16_CONSTANTS = std::to_array<std::uint32_t>({ sizeof(std::uint16_t), 128, 128, 32, 256, 1, 1 });

    // Argument definitions for dilated convolution
    const std::array<MLSSarg, 22> dilated_convolution_ARGS_CONSTANTS = {
        {{0, MLSS_FLOAT16, true, 2, true, true, false, "src"},
        {1, MLSS_FLOAT16, true, 2, true, true, false, "filter"},
        {2, MLSS_FLOAT16, true, 2, false, false, true, "dst"},
        {3, MLSS_FLOAT16, true, 2, true, true, false, "bias"},
        {4, MLSS_UINT32, false, 0, true, true, false, "groups"},
        {5, MLSS_UINT32, false, 0, true, true, false, "n"},
        {6, MLSS_UINT32, false, 0, true, true, false, "c"},
        {7, MLSS_UINT32, false, 0, true, true, false, "h"},
        {8, MLSS_UINT32, false, 0, true, true, false, "w"},
        {9, MLSS_UINT32, false, 0, true, true, false, "k"},
        {10, MLSS_UINT32, false, 0, true, true, false, "s"},
        {11, MLSS_UINT32, false, 0, true, true, false, "r"},
        {12, MLSS_UINT32, false, 0, true, true, false, "outW"},
        {13, MLSS_UINT32, false, 0, true, true, false, "outH"},
        {14, MLSS_UINT32, false, 0, true, true, false, "convStrideX"},
        {15, MLSS_UINT32, false, 0, true, true, false, "convStrideY"},
        {16, MLSS_UINT32, false, 0, true, true, false, "filterStrideX"},
        {17, MLSS_UINT32, false, 0, true, true, false, "filterStrideY"},
        {18, MLSS_UINT32, false, 0, true, true, false, "startPadX"},
        {19, MLSS_UINT32, false, 0, true, true, false, "startPadY"},
        {20, MLSS_UINT32, false, 0, true, true, false, "endPadX"},
        {21, MLSS_UINT32, false, 0, true, true, false, "endPadY"}
    };

} // namespace mlss::conv::dilated::hip::wmma::fp16
