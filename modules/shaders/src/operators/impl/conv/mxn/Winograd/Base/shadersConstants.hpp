/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::base::fp16
{

    struct WinogradHyperSet
    {
        // Each array is index by: isStride2 * 4 + isForward * 2 + isFp32 (see InsideHyperboloidSet).
        std::array<HyperConsts, 8> f3x3;
        std::array<HyperConsts, 8> f5x5;
        std::array<HyperConsts, 8> f7x7;
    };

    // And now the big list of ASIC-specific hyperboloids. Each is an equation of best fit computed from metacommand vs
// DirectML 1.8.0 shaders on data from the parameter cube:
// - c = 1 + range(4, 65, 4)
// - k = range(1, 16) + range(24, 267, 8)
// - x&y = range(1, 16) + range(24, 267, 8)
// If you change these, make sure they fit in 8 bits! There's a warning in the script to help catch it.
const WinogradHyperSet HyperSetNavi31 =
{
{{
    { 0, 0, 0, 114, }, // accuracy: 85.0%, 3x3, stride 1, backward, fp16
    { 0, 0, 0, 139, }, // accuracy: 79.3%, 3x3, stride 1, backward, fp32
    { 0, 0, 0, 181, }, // accuracy: 84.2%, 3x3, stride 1, forward,  fp16
    { 0, 0, 0, 221, }, // accuracy: 80.1%, 3x3, stride 1, forward,  fp32
    { 0, 0, 0, 137, }, // accuracy: 92.1%, 3x3, stride 2, backward, fp16
    { 3, 0, 0, 274, }, // accuracy: 87.3%, 3x3, stride 2, backward, fp32
    { 0, 0, 25, 161,}, // accuracy: 92.2%, 3x3, stride 2, forward,  fp16
    AlwaysFail,        // accuracy: 84.6%, 3x3, stride 2, forward,  fp32
}},
{{
    { 0, 0, 23, 48,}, // accuracy: 96.9%, 5x5, stride 1, backward, fp16
    { 0, 0, 23, 52,}, // accuracy: 97.2%, 5x5, stride 1, backward, fp32
    { 1, 0, 22, 56,}, // accuracy: 97.1%, 5x5, stride 1, forward,  fp16
    { 0, 0, 29, 73,}, // accuracy: 96.1%, 5x5, stride 1, forward,  fp32
    { 0, 0, 0,  66,}, // accuracy: 90.6%, 5x5, stride 2, backward, fp16
    { 0, 0, 0,  95,}, // accuracy: 94.2%, 5x5, stride 2, backward, fp32
    { 2, 0, 19, 49,}, // accuracy: 96.1%, 5x5, stride 2, forward,  fp16
    { 1, 0, 27, 48,}, // accuracy: 94.3%, 5x5, stride 2, forward,  fp32
}},
{{
    { 0, 0, 23, 47, }, // accuracy: 96.7%, 7x7, stride 1, backward, fp16
    { 0, 0, 23, 46, }, // accuracy: 96.9%, 7x7, stride 1, backward, fp32
    { 1, 0, 23, 57, }, // accuracy: 97.4%, 7x7, stride 1, forward,  fp16
    { 1, 0, 30, 64, }, // accuracy: 96.4%, 7x7, stride 1, forward,  fp32
    { 1, 0, 0,  77, }, // accuracy: 91.7%, 7x7, stride 2, backward, fp16
    { 0, 0, 7,  106,}, // accuracy: 95.1%, 7x7, stride 2, backward, fp32
    { 1, 0, 23, 57, }, // accuracy: 94.9%, 7x7, stride 2, forward,  fp16
    { 2, 0, 39, 44, }, // accuracy: 93.1, 7x7, stride 2, forward,  fp32
}},
};

const WinogradHyperSet HyperSetNavi32 =
{
{{
    { 0, 0, 0, 114, }, // accuracy: 85.0%, 3x3, stride 1, backward, fp16
    { 0, 0, 0, 139, }, // accuracy: 79.3%, 3x3, stride 1, backward, fp32
    { 0, 0, 0, 181, }, // accuracy: 84.2%, 3x3, stride 1, forward,  fp16
    { 0, 0, 0, 221, }, // accuracy: 80.1%, 3x3, stride 1, forward,  fp32
    { 0, 0, 0, 137, }, // accuracy: 92.1%, 3x3, stride 2, backward, fp16
    { 3, 0, 0, 274, }, // accuracy: 87.3%, 3x3, stride 2, backward, fp32
    { 0, 0, 25, 161,}, // accuracy: 92.2%, 3x3, stride 2, forward,  fp16
    AlwaysFail,        // accuracy: 84.6%, 3x3, stride 2, forward,  fp32
}},
{{
    { 0, 0, 14, 43,}, // accuracy: 95.2%, 5x5, stride 1, backward, fp16
    { 0, 0, 23, 52,}, // accuracy: 97.2%, 5x5, stride 1, backward, fp32
    { 1, 0, 14, 45,}, // accuracy: 95.6%, 5x5, stride 1, forward,  fp16
    { 0, 0, 29, 73,}, // accuracy: 96.1%, 5x5, stride 1, forward,  fp32
    { 0, 0, 0,  53,}, // accuracy: 91.3%, 5x5, stride 2, backward, fp16
    { 0, 0, 0,  89,}, // accuracy: 95.1%, 5x5, stride 2, backward, fp32
    { 1, 0, 15, 42,}, // accuracy: 95.5%, 5x5, stride 2, forward,  fp16
    { 1, 0, 23, 36,}, // accuracy: 94.7%, 5x5, stride 2, forward,  fp32
}},
{{
    { 0, 0, 14, 43, }, // accuracy: 95.8%, 7x7, stride 1, backward, fp16
    { 0, 0, 22, 50, }, // accuracy: 96.8%, 7x7, stride 1, backward, fp32
    { 1, 0, 12, 48, }, // accuracy: 94.9%, 7x7, stride 1, forward,  fp16
    { 1, 0, 23, 54, }, // accuracy: 96.3%, 7x7, stride 1, forward,  fp32
    { 0, 0, 0,  66, }, // accuracy: 93.6%, 7x7, stride 2, backward, fp16
    { 0, 0, 7,  106,}, // accuracy: 95.1%, 7x7, stride 2, backward, fp32
    { 1, 0, 23, 33, }, // accuracy: 94.7%, 7x7, stride 2, forward,  fp16
    { 1, 0, 27, 42, }, // accuracy: 93.8%, 7x7, stride 2, forward,  fp32
}},
};

const WinogradHyperSet HyperSetNavi33 =
{
{{
    { 0, 0, 0, 58,  }, // accuracy: 90.0%, 3x3, stride 1, backward, fp16
    { 0, 0, 0, 86,  }, // accuracy: 86.4%, 3x3, stride 1, backward, fp32
    { 0, 0, 0, 115, }, // accuracy: 89.0%, 3x3, stride 1, forward,  fp16
    { 0, 0, 0, 141, }, // accuracy: 87.0%, 3x3, stride 1, forward,  fp32
    { 0, 0, 0, 137, }, // accuracy: 92.1%, 3x3, stride 2, backward, fp16
    { 3, 0, 0, 274, }, // accuracy: 87.3%, 3x3, stride 2, backward, fp32
    { 0, 0, 25, 161,}, // accuracy: 92.2%, 3x3, stride 2, forward,  fp16
    AlwaysFail,        // accuracy: 84.6%, 3x3, stride 2, forward,  fp32
}},
{{
    { 0, 0, 12, 29,}, // accuracy: 92.0%, 5x5, stride 1, backward, fp16
    { 0, 0, 23, 52,}, // accuracy: 97.2%, 5x5, stride 1, backward, fp32
    { 0, 0, 13, 49,}, // accuracy: 93.5%, 5x5, stride 1, forward,  fp16
    { 0, 0, 29, 73,}, // accuracy: 96.1%, 5x5, stride 1, forward,  fp32
    { 0, 0, 0,  66,}, // accuracy: 90.6%, 5x5, stride 2, backward, fp16
    { 0, 0, 0,  60,}, // accuracy: 94.0%, 5x5, stride 2, backward, fp32
    { 1, 0, 15, 30,}, // accuracy: 94.0%, 5x5, stride 2, forward,  fp16
    { 1, 0, 23, 35,}, // accuracy: 94.4%, 5x5, stride 2, forward,  fp32
}},
{{
    { 0, 0, 23, 47, }, // accuracy: 96.7%, 7x7, stride 1, backward, fp16
    { 0, 0, 23, 46, }, // accuracy: 96.9%, 7x7, stride 1, backward, fp32
    { 0, 0, 23, 44, }, // accuracy: 96.1%, 7x7, stride 1, forward,  fp16
    { 1, 0, 30, 64, }, // accuracy: 96.4%, 7x7, stride 1, forward,  fp32
    { 1, 0, 0,  77, }, // accuracy: 91.7%, 7x7, stride 2, backward, fp16
    { 0, 0, 7,  106,}, // accuracy: 95.1%, 7x7, stride 2, backward, fp32
    { 0, 0, 24, 34, }, // accuracy: 94.5%, 7x7, stride 2, forward,  fp16
    { 0, 0, 32, 24, }, // accuracy: 93.0,  7x7, stride 2, forward,  fp32
}},
};

    const std::array<MLSSarg, 48> winograd_conv_ARGS_CONSTANTS = {{
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
        {40,  MLSS_UINT8,   false, 0, true,  true,  false, "activationMode"}, // offset 200
        {41,  MLSS_UINT8,   false, 0, true,  true,  false, "syncLimit"},      // offset 201  (MUST be 255)
        {42,  MLSS_UINT8,   false, 0, true,  true,  false, "syncPeriod"},     // offset 202
        {43,  MLSS_UINT8,   false, 0, false, false, false, "reserved8"},      // offset 203
        {44,  MLSS_UINT32,  false, 0, false, false, false, "reserved9"},      // offset 204
        {45,  MLSS_UINT64,  true,  1, true,  true,  false, "syncAddr"},       // offset 208 (device ptr)
        {46,  MLSS_UINT64,  true,  1, true,  true,  false, "accAddr"},        // offset 216 (device ptr)
        {47,  MLSS_UINT64,  false, 0, true,  true,  false, "aOffset"},        // offset 224
    }};

} // namespace mlss::conv::mxn::winograd::base::fp16
