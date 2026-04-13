/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::rage
{

    enum class ConvRageShader : std::uint32_t
    {
        Elf_Gfx12_460 = 0x00u,
        Elf_Gfx12_490 = 0x01u,
        Count
    };

    struct PerfModelCostRage
    {
        std::uint64_t constCost;
        std::uint64_t feCost;
        std::uint64_t phCost;
        std::uint64_t beCost;
    };

    constexpr std::uint64_t Gfx12MacRate  = 0x0200u;
    constexpr std::uint32_t MaxDispatches = 0x08u;

    constexpr PerfModelCostRage Gfx12ModelCost[static_cast<std::uint32_t>(ConvRageShader::Count)] =
    {
        { 0x2521u, 0x004Fu, 0x05F2u, 0x05FDu },
        { 0x260Cu, 0x00B6u, 0x05F2u, 0x05FDu }
    };

} // namespace mlss::conv::mxn::winograd::rage

namespace mlss::conv::mxn::winograd::rage::fp16
{

    const std::array<MLSSarg, 43> winograd_conv_ARGS_CONSTANTS = {{
        { 0,  MLSS_UINT32,  false, 0, true,  true,  false, "n"},
        { 1,  MLSS_UINT32,  false, 0, true,  true,  false, "c"},
        { 2,  MLSS_UINT32,  false, 0, true,  true,  false, "h"},
        { 3,  MLSS_UINT32,  false, 0, true,  true,  false, "w"},
        { 4,  MLSS_UINT32,  false, 0, true,  true,  false, "k"},
        { 5,  MLSS_UINT32,  false, 0, true,  true,  false, "nGroups"},
        { 6,  MLSS_UINT64,  false, 0, true,  true,  false, "flags64"},
        { 7,  MLSS_UINT64,  true,  1, true,  true,  false, "dataAddr"},
        { 8,  MLSS_UINT64,  true,  1, true,  true,  false, "filterAddr"},
        { 9,  MLSS_UINT64,  true,  1, false, false, true,  "outputAddr"},
        {10,  MLSS_UINT64,  false, 0, false, false, false, "reserved3"},
        {11,  MLSS_UINT32,  false, 0, true,  true,  false, "r"},
        {12,  MLSS_UINT32,  false, 0, true,  true,  false, "s"},
        {13,  MLSS_INT32,   false, 0, true,  true,  false, "padH"},
        {14,  MLSS_INT32,   false, 0, true,  true,  false, "padW"},
        {15,  MLSS_UINT32,  false, 0, true,  true,  false, "outH"},
        {16,  MLSS_UINT32,  false, 0, true,  true,  false, "outW"},
        {17,  MLSS_UINT64,  true,  1, true,  true,  false, "biasAddr"},
        {18,  MLSS_FLOAT32, false, 0, true,  true,  false, "alpha"},
        {19,  MLSS_FLOAT32, false, 0, true,  true,  false, "beta"},
        {20,  MLSS_UINT64,  false, 0, true,  true,  false, "dOffset"},
        {21,  MLSS_UINT64,  false, 0, true,  true,  false, "fOffset"},
        {22,  MLSS_UINT64,  false, 0, true,  true,  false, "oOffset"},
        {23,  MLSS_UINT64,  false, 0, true,  true,  false, "bOffset"},
        {24,  MLSS_UINT32,  false, 0, true,  true,  false, "dNStride"},
        {25,  MLSS_UINT32,  false, 0, true,  true,  false, "dCStride"},
        {26,  MLSS_UINT32,  false, 0, true,  true,  false, "dHStride"},
        {27,  MLSS_UINT32,  false, 0, false, false, false, "reserved4"},
        {28,  MLSS_UINT32,  false, 0, true,  true,  false, "fKStride"},
        {29,  MLSS_UINT32,  false, 0, true,  true,  false, "fCStride"},
        {30,  MLSS_UINT32,  false, 0, true,  true,  false, "fRStride"},
        {31,  MLSS_UINT32,  false, 0, false, false, false, "reserved5"},
        {32,  MLSS_UINT32,  false, 0, true,  true,  false, "oNStride"},
        {33,  MLSS_UINT32,  false, 0, true,  true,  false, "oKStride"},
        {34,  MLSS_UINT32,  false, 0, true,  true,  false, "oHStride"},
        {35,  MLSS_UINT32,  false, 0, false, false, false, "reserved6"},
        {36,  MLSS_UINT32,  false, 0, true,  true,  false, "G"},
        {37,  MLSS_UINT32,  false, 0, true,  true,  false, "dGStride"},
        {38,  MLSS_UINT32,  false, 0, true,  true,  false, "fGStride"},
        {39,  MLSS_UINT32,  false, 0, true,  true,  false, "oGStride"},
        {40,  MLSS_UINT8,   false, 0, true,  true,  false, "activationMode"},
        {41,  MLSS_UINT8,   false, 0, false, false, false, "reserved7"},
        {42,  MLSS_UINT16,  false, 0, false, false, false, "reserved8"},
    }};

} // namespace mlss::conv::mxn::winograd::rage::fp16

