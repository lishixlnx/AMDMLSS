/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    enum class DataTypeFlags : std::uint64_t
    {
        INT4       = MLSS_INT4,
        UINT4      = MLSS_UINT4,
        INT8       = MLSS_INT8,
        UINT8      = MLSS_UINT8,
        INT16      = MLSS_INT16,
        UINT16     = MLSS_UINT16,
        INT32      = MLSS_INT32,
        UINT32     = MLSS_UINT32,
        INT64      = MLSS_INT64,
        UINT64     = MLSS_UINT64,
        FLOAT4     = MLSS_FLOAT4,
        FLOAT8     = MLSS_FLOAT8,
        FLOAT8_FNUZ = MLSS_FLOAT8_FNUZ,
        FLOAT8_OCP = MLSS_FLOAT8_OCP,
        FLOAT16    = MLSS_FLOAT16,
        FLOAT32    = MLSS_FLOAT32,
        FLOAT64    = MLSS_FLOAT64,
        BFLOAT4    = MLSS_BFLOAT4,
        BFLOAT8    = MLSS_BFLOAT8,
        BFLOAT8_FNUZ = MLSS_BFLOAT8_FNUZ,
        BFLOAT8_OCP = MLSS_BFLOAT8_OCP,
        BFLOAT16   = MLSS_BFLOAT16,
    };

    //=====================================================================================================================
    enum class PrecisionFlags : std::uint32_t
    {
        FLOAT32             = MLSS_PRECISION_FLOAT32,
        FLOAT16             = MLSS_PRECISION_FLOAT16,
        FLOAT16_ADD_FLOAT32 = MLSS_PRECISION_FLOAT16_ADD_FLOAT32,
        COUNT               = MLSS_PRECISION_COUNT,
    };

    //=====================================================================================================================
    enum class ActivationFunctionFlags : std::uint32_t
    {
        ELU                 = MLSS_ACTIVATION_ELU,
        HARDMAX             = MLSS_ACTIVATION_HARDMAX,
        HARD_SIGMOID        = MLSS_ACTIVATION_HARD_SIGMOID,
        IDENTITY            = MLSS_ACTIVATION_IDENTITY,
        LEAKY_RELU          = MLSS_ACTIVATION_LEAKY_RELU,
        LINEAR              = MLSS_ACTIVATION_LINEAR,
        LOG_SOFTMAX         = MLSS_ACTIVATION_LOG_SOFTMAX,
        PARAMETERIZED_RELU  = MLSS_ACTIVATION_PARAMETERIZED_RELU,
        PARAMETRIC_SOFTPLUS = MLSS_ACTIVATION_PARAMETRIC_SOFTPLUS,
        RELU                = MLSS_ACTIVATION_RELU,
        SCALED_ELU          = MLSS_ACTIVATION_SCALED_ELU,
        SCALED_TANH         = MLSS_ACTIVATION_SCALED_TANH,
        SIGMOID             = MLSS_ACTIVATION_SIGMOID,
        SOFTMAX             = MLSS_ACTIVATION_SOFTMAX,
        SOFTPLUS            = MLSS_ACTIVATION_SOFTPLUS,
        SOFTSIGN            = MLSS_ACTIVATION_SOFTSIGN,
        TANH                = MLSS_ACTIVATION_TANH,
        THRESHOLDED_RELU    = MLSS_ACTIVATION_THRESHOLDED_RELU,
        COUNT                = MLSS_ACTIVATION_COUNT,
    };

    //=====================================================================================================================
    enum class OperatorFlag : std::uint32_t
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
    enum class GpuCodenameFlags : std::uint32_t
    {
        Tahiti,
        Pitcairn,
        Verde,
        Oland,
        Hainan,
        Kaveri,
        Spooky,
        Spectre,
        HawaiiPro,
        Grenada,
        Hawaii,
        Kabini,
        Mullins,
        Bonaire,
        Godavari,
        Gladius,
        Carrizo,
        BristolRidge,
        Iceland,
        Tonga,
        Fiji,
        Polaris10,
        Polaris11,
        Polaris12,
        Polaris22,
        TongaPro,
        StoneyRidge,
        Vega10,
        Greenland,
        RavenRidge,
        Vega12,
        Vega20,
        Arcturus,
        MI100,
        RavenRidge2,
        MI200,
        Aldebaran,
        Renoir,
        MI300A,
        MI300X,
        MI325X,
        AquaVanjaram,
        MI350,
        Ariel,
        Navi10Lite,
        Navi10,
        Navi12,
        Navi14,
        Robin,
        Arden,
        Navi21Lite,
        Navi21,
        SiennaCichlid,
        Navi22,
        NavyFlounder,
        Navi23,
        DimgrayCaveFish,
        VanGogh,
        Mero,
        Navi24,
        Rembrandt,
        Raphael,
        Mendocino,
        Viola,
        Navi31,
        Navi32,
        Navi33,
        Phoenix,
        Phoenix2,
        Navi32GLXL,
        Navi3_5,
        Sarlak,
        Krackan2e,
        GorgonPoint3,
        Strix1A0,
        Strix1,
        Strix2,
        Strix3,
        StrixHalo,
        GorgonPoint1,
        Krackan,
        GorgonPoint2,
        Medusa3,
        Medusa1A0,
        Medusa1B0,
        Medusa2,
        Navi44,
        Navi48,
        MI400XCDML,
        MI400XCDGP,
        Navi44A0,
        Navi48A0,
        Canis,
        Magnus,
        Orion,
        Mariner,
        VanGoghLite,
        Mobile0,
        Mobile1,
        Viking,
        MGFX1,
        Mobile2EVT1plus,
        Mobile3,
        AlphaTrion0,
        AlphaTrion1,
        AlphaTrion2,
        AlphaTrion3,
        Unknown
    };

    enum class AsicsTypesFlags : std::uint32_t
    {
        APU,
        dGPU,
        gGPU,
        unknown
    };

    enum class BinaryTypeFlags : std::uint32_t
    {
        ELF  = 0x01u,
        DXIL = 0x02u,
    };

    //=====================================================================================================================
    enum class ShaderTypesFlags : std::uint64_t
    {
        UNKNOWN = 1 << 0,
        WMMA = 1 << 1,
        XDL = 1 << 2,
        DL = 1 << 3,
        SPIRV = 1 << 4,
        HLSL = 1 << 5,
    };

    ShaderTypesFlags operator|(ShaderTypesFlags lhs, ShaderTypesFlags rhs);
    ShaderTypesFlags operator&(ShaderTypesFlags lhs, ShaderTypesFlags rhs);
    ShaderTypesFlags operator^(ShaderTypesFlags lhs, ShaderTypesFlags rhs);
    ShaderTypesFlags operator~(ShaderTypesFlags rhs);
    ShaderTypesFlags operator&=(ShaderTypesFlags& lhs, ShaderTypesFlags rhs);
    ShaderTypesFlags operator|=(ShaderTypesFlags& lhs, ShaderTypesFlags rhs);
    ShaderTypesFlags operator^=(ShaderTypesFlags& lhs, ShaderTypesFlags rhs);

    bool operator==(ShaderTypesFlags lhs, ShaderTypesFlags rhs);
    bool operator!=(ShaderTypesFlags lhs, ShaderTypesFlags rhs);
    


} // namespace mlss
