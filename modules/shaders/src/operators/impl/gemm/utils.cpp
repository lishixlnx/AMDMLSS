/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "utils.hpp"

namespace mlss::gemm::utils
{

    namespace
    {

        constexpr std::size_t kHyperSetSize = 4u;

        // Hyperboloid tunings for the GEMM HIP MxN backend.
        // Index = hasC + transB * 2 (matches dxcp::ChooseTuning).
        inline constexpr std::array<HyperConsts, kHyperSetSize> HyperSetNavi31 = {
            HyperConsts{ 0u, 100u,  5u,   49u},
            HyperConsts{ 0u,  25u,  0u,    1u},
            HyperConsts{ 0u,  36u, 15u, 1552u},
            HyperConsts{ 0u,  49u, 34u, 1187u}
        };

        inline constexpr std::array<HyperConsts, kHyperSetSize> HyperSetNavi32 = {
            HyperConsts{ 0u,  15u,  3u,   76u},
            HyperConsts{ 0u,  15u,  3u,   19u},
            HyperConsts{ 0u,  36u, 15u, 1552u},
            HyperConsts{ 0u,  49u, 34u, 1187u}
        };

        inline constexpr std::array<HyperConsts, kHyperSetSize> HyperSetNavi33 = {
            HyperConsts{ 0u,  12u,  3u,   31u},
            HyperConsts{ 0u,   0u,  0u,    1u},
            HyperConsts{ 0u,   0u, 32u,  745u},
            HyperConsts{ 0u,   0u,  0u,    1u}
        };

        inline constexpr std::array<HyperConsts, kHyperSetSize> HyperSetPhoenix = {
            HyperConsts{ 0u,  12u,  3u,   27u},
            HyperConsts{ 0u,   0u,  0u,    1u},
            HyperConsts{ 0u,   0u, 28u,  492u},
            HyperConsts{ 0u,   0u,  0u,    1u}
        };

    } // namespace

    GenericGemmParams buildGemmParams(const std::vector<Attribute>& attributes)
    {
        GenericGemmParams params{};
        params.batch      = 1u;
        params.alpha      = 1.0f;
        params.beta       = 0.0f;
        params.precision  = PrecisionFlags::COUNT;
        params.activation = ActivationFunctionFlags::COUNT;

        for (const auto& attribute : attributes)
        {
            if (attribute.is(MLSS_ATTR_GEMM_M))
            {
                params.m = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_N))
            {
                params.n = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_K))
            {
                params.k = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_BATCH))
            {
                params.batch = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_ALPHA))
            {
                params.alpha = attribute.value<float>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_BETA))
            {
                params.beta = attribute.value<float>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_HASC))
            {
                params.hasC = attribute.value<bool>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_TRANSA))
            {
                params.transA = attribute.value<bool>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_TRANSB))
            {
                params.transB = attribute.value<bool>();
            }
            else if (attribute.is(MLSS_ATTR_GEMM_DATATYPE))
            {
                params.dataType = static_cast<DataTypeFlags>(attribute.value<std::uint32_t>());
            }
            else if (attribute.is(MLSS_ATTR_GEMM_PRECISION))
            {
                params.precision = static_cast<PrecisionFlags>(attribute.value<std::uint32_t>());
            }
            else if (attribute.is(MLSS_ATTR_GEMM_ACTIVATION))
            {
                params.activation = static_cast<ActivationFunctionFlags>(attribute.value<std::uint32_t>());
            }
        }

        return params;
    }

    bool chooseTuning(const GfxIpTriple& gfxip, const GenericGemmParams& params)
    {
        const std::size_t mode = static_cast<std::size_t>(params.hasC)
                               + 2u * static_cast<std::size_t>(params.transB);

        if (mode >= kHyperSetSize)
        {
            return false;
        }

        if (gfxip == IP_GFX1100)
        {
            // Navi31 always wins at higher batch numbers.
            if (params.batch > 1u)
            {
                return true;
            }
            return calcHyperboloid(HyperSetNavi31[mode],
                                   static_cast<float>(params.m),
                                   static_cast<float>(params.n),
                                   static_cast<float>(params.k));
        }

        if (gfxip == IP_GFX1101)
        {
            return calcHyperboloid(HyperSetNavi32[mode],
                                   static_cast<float>(params.m),
                                   static_cast<float>(params.n),
                                   static_cast<float>(params.k));
        }

        if (gfxip == IP_GFX1102 || gfxip == IP_GFX1103)
        {
            return calcHyperboloid(HyperSetNavi33[mode],
                                   static_cast<float>(params.m),
                                   static_cast<float>(params.n),
                                   static_cast<float>(params.k));
        }

        if (isGfx115x(gfxip))
        {
            return calcHyperboloid(HyperSetPhoenix[mode],
                                   static_cast<float>(params.m),
                                   static_cast<float>(params.n),
                                   static_cast<float>(params.k));
        }

        if (isGfx120x(gfxip))
        {
            // Navi44 / Navi48 always supported until performance data is collected.
            return true;
        }

        return false;
    }

} // namespace mlss::gemm::utils
