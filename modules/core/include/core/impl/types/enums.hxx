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
    enum class GpuCodenameFlags
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

    enum class AsicsTypesFlags
    {
        APU,
        dGPU,
        gGPU,
        unknown
    };

    enum class ShaderTypesFlags
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
