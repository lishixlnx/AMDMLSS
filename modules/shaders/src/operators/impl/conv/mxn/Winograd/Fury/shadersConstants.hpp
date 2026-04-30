/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::fury
{

    enum class ConvFuryShader : std::uint32_t
    {
        Elf_Gfx11_C16 = 0x00u,
        Elf_Gfx11_C32 = 0x01u,
        Elf_Gfx11_Navi33_C16 = 0x02u,
        Elf_Gfx12_C16 = 0x03u,
        Elf_Gfx12_C32 = 0x04u,
        Count
    };

    enum ConvAccumMode : std::uint32_t
    {
        ConvAccumModeC16 = 0x00u,
        ConvAccumModeC32 = 0x01u,
        ConvAccumModeCount
    };

    struct PerfModelCostFury
    {
        std::uint64_t startCost;
        std::uint64_t accumCost;
        std::uint64_t activCost;
        std::uint64_t filterCost;
    };

    constexpr std::uint64_t Gfx11MacRate = 0x0100u;
    constexpr std::uint64_t Gfx12MacRate = 0x0200u;

    constexpr PerfModelCostFury Gfx11ModelCosts[ConvAccumModeCount] =
    {
        { 0x05AAu, 0x066Du, 0x06A0u, 0x05AAu },
        { 0x0A28u, 0x0B7Au, 0x0BADu, 0x0A28u }
    };

    constexpr PerfModelCostFury Gfx12ModelCosts[ConvAccumModeCount] =
    {
        { 0x03F2u, 0x053Fu, 0x05D7u, 0x03F2u },
        { 0x06D2u, 0x08EFu, 0x098Bu, 0x06D2u }
    };

} // namespace mlss::conv::mxn::winograd::fury

namespace mlss::conv::mxn::winograd::fury::fp16
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

} // namespace mlss::conv::mxn::winograd::fury::fp16
