/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1201/fp16/shadersBinReloc.hpp"
#include "gfx1201/fp16/shadersBinNonReloc.hpp"

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::mxn::winograd::rage
{

    using namespace mlss::conv::mxn::winograd::rage::fp16;

    namespace
    {

        template <class T>
            requires std::is_integral_v<T>
        T roundUpToMultiple(const T& value, const std::type_identity_t<T>& multiple)
        {
            return integer_divide_ceil(value, multiple) * multiple;
        }

        // See Base/shadersUtils.cpp for the rationale: the non-relocatable
        // ELFs are pre-linked offline by tools/winograd_pal/ because comgr's
        // LINK_RELOCATABLE_TO_EXECUTABLE silently exits on PAL OS/ABI inputs.
        struct ShaderPair
        {
            ShaderDescriptorType reloc;
            ShaderDescriptorType nonReloc;
        };

#define MLSS_WINOGRAD_RAGE_PAIR(NS, NAME)                                      \
    ShaderPair                                                                 \
    {                                                                          \
        .reloc    = make_shader_descriptor(NS::NAME),                          \
        .nonReloc = make_shader_descriptor(NS::NAME##_NonReloc),               \
    }

        //=============================================================================================================
        ConvRageShader selectShaderVariant(const GenericConvParams& params)
        {
            if ((params.r <= 0x03u) && (params.s <= 0x03u))
            {
                return ConvRageShader::Elf_Gfx12_460;
            }
            return ConvRageShader::Elf_Gfx12_490;
        }

        //=============================================================================================================
        ShaderPair selectShaderPair(const GfxIpTriple& gfxip, ConvRageShader shader)
        {
            if (isGfx120x(gfxip))
            {
                if (shader == ConvRageShader::Elf_Gfx12_460)
                {
                    return MLSS_WINOGRAD_RAGE_PAIR(fp16::gfx1201, ConvRage_460_Navi48_Fp16_F2x3_Stride1_Elf);
                }
                return MLSS_WINOGRAD_RAGE_PAIR(fp16::gfx1201, ConvRage_490_Navi48_Fp16_F2x3_Stride1_Elf);
            }
            return {};
        }

        //=============================================================================================================
        std::uint32_t computeBestNGroups(
            const GenericConvParams& params,
            std::uint32_t cuCount,
            ConvRageShader shader)
        {
            constexpr std::uint64_t tR  = 0x03u;
            constexpr std::uint64_t tS  = 0x03u;
            constexpr std::uint64_t tOh = 0x02u;
            constexpr std::uint64_t tOw = 0x02u;

            constexpr std::uint64_t nhwFactor  = 0x3Eu;
            constexpr std::uint64_t kFactor    = 0x20u;
            constexpr std::uint64_t cFactor    = 0x10u;
            const std::uint64_t nhwFactorG = roundUpToMultiple(nhwFactor, std::uint64_t{0x20u});

            const std::uint64_t s  = params.s;
            const std::uint64_t r  = params.r;
            const std::uint64_t oW = params.outW;
            const std::uint64_t oH = params.outH;
            const std::uint64_t n  = params.n;
            const std::uint64_t c  = static_cast<std::uint64_t>(params.c) / params.groups;
            const std::uint64_t k  = static_cast<std::uint64_t>(params.k) / params.groups;
            const std::uint64_t g  = params.groups;

            const std::uint64_t nkhwPerWork = kFactor * nhwFactorG * tOh * tOw;

            const std::uint64_t Rg  = roundUpToMultiple(r, tR);
            const std::uint64_t Sg  = roundUpToMultiple(s, tS);
            const std::uint64_t Cg  = roundUpToMultiple(c, cFactor);
            const std::uint64_t Kg  = roundUpToMultiple(k, kFactor);
            const std::uint64_t oHg = roundUpToMultiple(oH, tOh);
            const std::uint64_t oWg = roundUpToMultiple(oW, tOw) + tOw;

            const std::uint64_t sLoops   = Sg / tS;
            const std::uint64_t rLoops   = Rg / tR;
            const std::uint64_t cLoops   = Cg / cFactor;
            const std::uint64_t kGroups  = Kg / kFactor;
            const std::uint64_t nhwTiles = n * integer_divide_ceil(oHg, tOh) * integer_divide_ceil(oWg, tOw);
            const std::uint64_t macs     = n * g * k * c * oH * r * oW * s;

            const auto& perfCost = Gfx12ModelCost[static_cast<std::uint32_t>(shader)];

            std::uint32_t bestNGroups = 0x00u;
            float bestWti = 0.0f;

            for (std::uint32_t nDisp = 0x01u; nDisp <= MaxDispatches; ++nDisp)
            {
                const std::uint64_t nGroups = integer_divide_ceil(static_cast<std::uint64_t>(cuCount),
                                                                  g) * nDisp;
                if (nGroups < kGroups)
                {
                    continue;
                }

                const std::uint64_t nGroupsE    = kGroups * (nGroups / kGroups);
                const std::uint64_t nWorks       = kGroups * integer_divide_ceil(nhwTiles, nhwFactor);
                const std::uint64_t nDispatches  = integer_divide_ceil(g * nGroupsE,
                                                                       static_cast<std::uint64_t>(cuCount));
                const std::uint64_t nWorksPerCu  = integer_divide_ceil(nWorks, nGroupsE);

                const std::uint64_t nConsts = nDispatches;
                const std::uint64_t feCalls = nDispatches * nWorksPerCu * sLoops * rLoops;
                const std::uint64_t phCalls = nDispatches * nWorksPerCu * sLoops * rLoops * cLoops;
                const std::uint64_t beCalls = nDispatches * nWorksPerCu;

                const std::uint64_t predictedClk = nConsts * perfCost.constCost
                                                 + feCalls * perfCost.feCost
                                                 + phCalls * perfCost.phCost
                                                 + beCalls * perfCost.beCost;

                if (predictedClk == 0x00u)
                {
                    continue;
                }

                const float idealDirectClk = static_cast<float>(macs)
                                           / static_cast<float>(Gfx12MacRate)
                                           / static_cast<float>(cuCount);

                const float wti = idealDirectClk / static_cast<float>(predictedClk);

                if (wti > bestWti)
                {
                    bestWti = wti;
                    bestNGroups = static_cast<std::uint32_t>(nGroups);
                }
            }

            return bestNGroups;
        }

    } // namespace

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const GenericConvParams& params)
    {
        using mlss::op::utils::MetaCmdCaps;

        bool isArchSupported = isGfx120x(gfxip);
        bool isFp16 = params.dataType == DataTypeFlags::FLOAT16;

        bool isSupported = isArchSupported && isFp16;

        if (isSupported)
        {
            bool isStride1 = (params.inputStrideX == 0x01u) && (params.convStrideX == 0x01u)
                          && (params.inputStrideY == 0x01u) && (params.convStrideY == 0x01u)
                          && (params.filterStrideX == 0x01u) && (params.filterStrideY == 0x01u);

            isSupported = isStride1;
        }

        if (isSupported)
        {
            constexpr std::uint64_t Size256Mb = 0x10000000u;

            const std::uint64_t k = static_cast<std::uint64_t>(params.k) / params.groups;
            const std::uint64_t c = static_cast<std::uint64_t>(params.c);

            isSupported = (((k - 0x01u) * c + 0x01u) * params.r * params.s < Size256Mb);
        }

        if (isSupported)
        {
            auto numCuResult = getNumCu(gfxip);
            if (numCuResult.has_value())
            {
                const auto numCu = static_cast<std::uint32_t>(numCuResult.value());
                const std::uint64_t k = static_cast<std::uint64_t>(params.k) / params.groups;
                const std::uint64_t kGroups = integer_divide_ceil(k, std::uint64_t{0x20u});
                const std::uint64_t maxNGroups = static_cast<std::uint64_t>(MaxDispatches)
                                               * integer_divide_ceil(static_cast<std::uint64_t>(numCu),
                                                                     static_cast<std::uint64_t>(params.groups));

                isSupported = (maxNGroups <= std::numeric_limits<std::uint16_t>::max())
                           && (maxNGroups >= kGroups);
            }
            else
            {
                isSupported = false;
            }
        }

        if (isSupported)
        {
            if (((params.groups > 0x01u) && (params.groups == params.c)) ||
                ((params.groups > 0x01u) && (params.groups == params.k)))
            {
                isSupported = false;
            }
        }

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isSupported ? 0x00000001u : 0x00000000u;
        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const GenericConvParams& params)
    {
        auto capsResult = isShadersAvailable(gfxip, params);
        if (capsResult.support == 0x00000000u)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
        }

        const auto shader = selectShaderVariant(params);

        const auto shaderPair = selectShaderPair(gfxip, shader);
        if (shaderPair.reloc.m_binary.empty() || shaderPair.nonReloc.m_binary.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        auto numCuResult = getNumCu(gfxip);
        if (!numCuResult.has_value())
        {
            return std::unexpected(numCuResult.error());
        }

        const auto numCu = static_cast<std::uint32_t>(numCuResult.value());
        const auto nGroups = computeBestNGroups(params, numCu, shader);

        auto workgroupSizeResult = getWorkgroupSize(shaderPair.reloc.m_binary);
        MLSSdim3 blocks = workgroupSizeResult.value_or(MLSSdim3{ 0x01u, 0x01u, 0x01u });
        MLSSdim3 grid{ nGroups * params.groups, 0x01u, 0x01u };

        Binaries binaries;

        Blob relocBlob = std::move(*make_binary_blob(shaderPair.reloc));
        relocBlob = fp16::winograd_conv_ARGS_CONSTANTS;
        relocBlob.setGridBlocks(grid, blocks);
        binaries.addBlob(std::move(relocBlob));

        Blob nonRelocBlob = std::move(*make_binary_blob(shaderPair.nonReloc));
        nonRelocBlob = fp16::winograd_conv_ARGS_CONSTANTS;
        nonRelocBlob.setGridBlocks(grid, blocks);
        binaries.addBlob(std::move(nonRelocBlob));

        return binaries;
    }

} // namespace mlss::conv::mxn::winograd::rage
