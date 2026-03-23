#pragma once

namespace mlss
{
    //=====================================================================================================================
    enum class DataTypeFlags : std::uint32_t
    {
        INT4,
        UINT4,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        INT64,
        UINT64,
        FLOAT4,
        FLOAT8,
        FLOAT16,
        FLOAT32,
        FLOAT64,
        BFLOAT4,
        BFLOAT8,
        BFLOAT16
    };

    //=====================================================================================================================
    enum class MCDPrecisionFlags : uint64_t
    {
        FLOAT32,
        FLOAT16,
        FLOAT16_ADD_FLOAT32,

        COUNT,
    };

    //=====================================================================================================================
    enum class MCDActivationFunctionFlags : uint64_t
    {
        ELU,
        HARDMAX,
        HARD_SIGMOID,
        IDENTITY,
        LEAKY_RELU,
        LINEAR,
        LOG_SOFTMAX,
        PARAMETERIZED_RELU,
        PARAMETRIC_SOFTPLUS,
        RELU,
        SCALED_ELU,
        SCALED_TANH,
        SIGMOID,
        SOFTMAX,
        SOFTPLUS,
        SOFTSIGN,
        TANH,
        THRESHOLDED_RELU,

        COUNT,
    };

    //=====================================================================================================================
    enum class OperatorFlag : uint64_t
    {
        GEMM,
        QUANTIZED_GEMM,
        CONV,
        CONV_DILATED,
        MVN,
        MHA,
        GQA,
        UNKNOWN_OP,
        COUNT = UNKNOWN_OP
    };

    //=====================================================================================================================
    enum class GfxArchitectureFlags
    {
        Gfx1010,
        Gfx1011,
        Gfx1012,
        Gfx1030,
        Gfx1031,
        Gfx1032,
        Gfx1034,
        Gfx1100,
        Gfx1101,
        Gfx1102,
        Gfx1103,
        Gfx1150,
        Gfx1151,
        Gfx1152,
        Gfx1153,
        Gfx1154,
        Gfx1170,
        Gfx1171,
        Gfx1200,
        Gfx1201,
        Unknown
    };

    //=====================================================================================================================
    enum class GpuCodenameFlags
    {
        Navi10,
        Navi11,
        Navi14,
        Navi21,
        Navi22,
        Navi23,
        Navi24,
        Navi31,
        Navi32,
        Navi33,
        Navi34,
        Strix1,
        StrixHalo,
        GorgonPoint1,
        Krackan,
        GorgonPoint2,
        Krackan2,
        Medusa3,
        Medusa1,
        Medusa2,
        Navi44,
        Navi48,
        Unknown
    };

} // mlss
