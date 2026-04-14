/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::conv::mxn::misa::fp16
{
    constexpr std::uint32_t VectorC = 0x08u;

    constexpr std::array<std::uint32_t, 3> MisaMxNBiasReluConsts = { 0x20u, 0x40u, 0x20u };
    constexpr std::array<std::uint32_t, 3> MisaMxNReluConsts     = { 0x20u, 0x40u, 0x20u };
    constexpr std::array<std::uint32_t, 3> MisaMxNBiasConsts     = { 0x40u, 0x20u, 0x20u };
    constexpr std::array<std::uint32_t, 3> MisaMxNConsts         = { 0x20u, 0x40u, 0x20u };

    const std::array<MLSSarg, 33> misa_conv_ARGS_CONSTANTS = {{
        { 0, MLSS_FLOAT16, true,  2, true,  true,  false, "pIn"},
        { 1, MLSS_FLOAT16, true,  2, true,  true,  false, "pWei"},
        { 2, MLSS_FLOAT16, true,  2, false, false, true,  "pOut"},
        { 3, MLSS_FLOAT16, true,  2, true,  true,  false, "pBias"},
        { 4, MLSS_UINT32,  false, 0, true,  true,  false, "hi"},
        { 5, MLSS_UINT32,  false, 0, true,  true,  false, "wi"},
        { 6, MLSS_UINT32,  false, 0, true,  true,  false, "n"},
        { 7, MLSS_UINT32,  false, 0, true,  true,  false, "k"},
        { 8, MLSS_UINT32,  false, 0, true,  true,  false, "c"},
        { 9, MLSS_UINT32,  false, 0, true,  true,  false, "g"},
        {10, MLSS_UINT32,  false, 0, true,  true,  false, "ho"},
        {11, MLSS_UINT32,  false, 0, true,  true,  false, "wo"},
        {12, MLSS_UINT32,  false, 0, true,  true,  false, "inStrideN"},
        {13, MLSS_UINT32,  false, 0, true,  true,  false, "inStrideC"},
        {14, MLSS_UINT32,  false, 0, true,  true,  false, "inStrideH"},
        {15, MLSS_UINT32,  false, 0, true,  true,  false, "inStrideW"},
        {16, MLSS_UINT32,  false, 0, true,  true,  false, "outStrideN"},
        {17, MLSS_UINT32,  false, 0, true,  true,  false, "outStrideK"},
        {18, MLSS_UINT32,  false, 0, true,  true,  false, "outStrideH"},
        {19, MLSS_UINT32,  false, 0, true,  true,  false, "outStrideW"},
        {20, MLSS_UINT32,  false, 0, true,  true,  false, "strideHw"},
        {21, MLSS_UINT32,  false, 0, true,  true,  false, "dilationHw"},
        {22, MLSS_UINT32,  false, 0, true,  true,  false, "padHw"},
        {23, MLSS_UINT32,  false, 0, true,  true,  false, "weiHw"},
        {24, MLSS_UINT32,  false, 0, true,  true,  false, "moveSliceK"},
        {25, MLSS_UINT32,  false, 0, true,  true,  false, "magic0"},
        {26, MLSS_UINT32,  false, 0, true,  true,  false, "magic1"},
        {27, MLSS_UINT32,  false, 0, true,  true,  false, "magic2"},
        {28, MLSS_UINT32,  false, 0, true,  true,  false, "magic3"},
        {29, MLSS_UINT32,  false, 0, true,  true,  false, "magic4"},
        {30, MLSS_UINT32,  false, 0, true,  true,  false, "magic5"},
        {31, MLSS_UINT32,  false, 0, true,  true,  false, "shiftPack0"},
        {32, MLSS_UINT32,  false, 0, true,  true,  false, "shiftPack1"}
    }};

} // namespace mlss::conv::mxn::misa::fp16
