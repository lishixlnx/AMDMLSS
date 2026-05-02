/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1100/fp16/shadersBinReloc.hpp"
#include "gfx1100/fp16/shadersBinNonReloc.hpp"
#include "gfx1201/fp16/shadersBinReloc.hpp"
#include "gfx1201/fp16/shadersBinNonReloc.hpp"

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::mxn::winograd::fury
{

    using namespace mlss::conv::mxn::winograd::fury::fp16;

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

#define MLSS_WINOGRAD_FURY_PAIR(NS, NAME)                                      \
    ShaderPair                                                                 \
    {                                                                          \
        .reloc    = make_shader_descriptor(NS::NAME),                          \
        .nonReloc = make_shader_descriptor(NS::NAME##_NonReloc),               \
    }

        //=============================================================================================================
        float computePerfWti(
            const GenericConvParams& params,
            std::uint32_t cuCount,
            bool isGfx12,
            bool c32Mode)
        {
            constexpr std::uint64_t tR  = 0x03u;
            constexpr std::uint64_t tS  = 0x03u;
            constexpr std::uint64_t tOh = 0x02u;
            constexpr std::uint64_t tOw = 0x02u;

            constexpr std::uint64_t nhwFactor  = 0x3Eu;
            constexpr std::uint64_t kFactor    = 0x10u;
            const std::uint64_t nhwFactorG = roundUpToMultiple(nhwFactor, std::uint64_t{0x20u});

            const std::uint64_t cFactor = c32Mode ? 0x20u : 0x10u;

            const std::uint64_t s  = params.s;
            const std::uint64_t r  = params.r;
            const std::uint64_t oW = params.outW;
            const std::uint64_t oH = params.outH;
            const std::uint64_t n  = params.n;
            const std::uint64_t c  = params.c;
            const std::uint64_t k  = params.k;
            const std::uint64_t g  = params.groups;

            const std::uint64_t Rg  = roundUpToMultiple(r, tR);
            const std::uint64_t Sg  = roundUpToMultiple(s, tS);
            const std::uint64_t Cg  = roundUpToMultiple(c, cFactor);
            const std::uint64_t Kg  = roundUpToMultiple(k, kFactor);
            const std::uint64_t oHg = roundUpToMultiple(oH, tOh);
            const std::uint64_t oWg = roundUpToMultiple(oW, tOw) + tOw;

            const std::uint64_t cLoops   = Cg / cFactor;
            const std::uint64_t kGroups  = Kg / kFactor;
            const std::uint64_t nGroups  = cuCount;
            const std::uint64_t nGroupsE = kGroups * (nGroups / kGroups);

            const std::uint64_t nkhwPerWork = kFactor * nhwFactorG * tOh * tOw;
            const std::uint64_t nhwTiles    = n * integer_divide_ceil(oHg, tOh) * integer_divide_ceil(oWg, tOw);
            const std::uint64_t nWorks      = kGroups * integer_divide_ceil(nhwTiles, nhwFactor);
            const std::uint64_t numDisp     = integer_divide_ceil(nGroupsE, static_cast<std::uint64_t>(cuCount));
            const std::uint64_t nWorksPerCu = integer_divide_ceil(nWorks, nGroupsE);

            const std::uint64_t macs = n * g * k * c * oH * r * oW * s;

            const auto& perfCost = isGfx12 ? Gfx12ModelCosts[c32Mode ? ConvAccumModeC32 : ConvAccumModeC16]
                                           : Gfx11ModelCosts[c32Mode ? ConvAccumModeC32 : ConvAccumModeC16];

            constexpr std::uint64_t nWorksPerFilter = 0x0Au;
            const std::uint64_t fReloads = (cLoops == 0x01u)
                ? 0x01u
                : integer_divide_ceil(nWorksPerCu, nWorksPerFilter);

            const std::uint64_t phStart  = numDisp * (c32Mode ? 0x04u : 0x06u);
            const std::uint64_t phAccum  = numDisp * nWorksPerCu * (cLoops - 0x01u);
            const std::uint64_t phActiv  = numDisp * nWorksPerCu;
            const std::uint64_t phFilter = numDisp * fReloads * cLoops;

            const std::uint64_t predictedClk = phStart  * perfCost.startCost
                                             + phAccum  * perfCost.accumCost
                                             + phActiv  * perfCost.activCost
                                             + phFilter * perfCost.filterCost;

            if (predictedClk == 0x00u)
            {
                return 0.0f;
            }

            const std::uint64_t macRate = isGfx12 ? Gfx12MacRate : Gfx11MacRate;
            const float idealDirectClk = static_cast<float>(macs)
                                       / static_cast<float>(macRate)
                                       / static_cast<float>(cuCount);

            return idealDirectClk / static_cast<float>(predictedClk);
        }

        //=============================================================================================================
        bool isReducedVgprAsic(const GfxIpTriple& gfxip)
        {
            return (gfxip == IP_GFX1101);
        }

        //=============================================================================================================
        ConvFuryShader selectShaderVariant(
            const GfxIpTriple& gfxip,
            const GenericConvParams& params,
            std::uint32_t numCu)
        {
            if (isGfx110x(gfxip) && isReducedVgprAsic(gfxip))
            {
                return ConvFuryShader::Elf_Gfx11_Navi33_C16;
            }

            const bool isGfx12 = isGfx120x(gfxip);

            const float wtiC16 = computePerfWti(params, numCu, isGfx12, false);
            const float wtiC32 = computePerfWti(params, numCu, isGfx12, true);

            if (isGfx12)
            {
                return (wtiC32 > wtiC16) ? ConvFuryShader::Elf_Gfx12_C32
                                         : ConvFuryShader::Elf_Gfx12_C16;
            }

            return (wtiC32 > wtiC16) ? ConvFuryShader::Elf_Gfx11_C32
                                     : ConvFuryShader::Elf_Gfx11_C16;
        }

        //=============================================================================================================
        ShaderPair selectShaderPair(ConvFuryShader shader)
        {
            switch (shader)
            {
                case ConvFuryShader::Elf_Gfx11_C16:
                    return MLSS_WINOGRAD_FURY_PAIR(fp16::gfx1100, ConvFury_Navi31_F2x3_C16_Stride1_Elf);
                case ConvFuryShader::Elf_Gfx11_C32:
                    return MLSS_WINOGRAD_FURY_PAIR(fp16::gfx1100, ConvFury_Navi31_F2x3_C32_Stride1_Elf);
                case ConvFuryShader::Elf_Gfx11_Navi33_C16:
                    return MLSS_WINOGRAD_FURY_PAIR(fp16::gfx1100, ConvFury_Navi33_F2x3_C16_Stride1_Elf);
                case ConvFuryShader::Elf_Gfx12_C16:
                    return MLSS_WINOGRAD_FURY_PAIR(fp16::gfx1201, ConvFury_Navi48_F2x3_C16_Stride1_Elf);
                case ConvFuryShader::Elf_Gfx12_C32:
                    return MLSS_WINOGRAD_FURY_PAIR(fp16::gfx1201, ConvFury_Navi48_F2x3_C32_Stride1_Elf);
                default:
                    return {};
            }
        }

    } // namespace

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        using mlss::op::utils::MetaCmdCaps;

        const auto params = cstmStruct
            ? *static_cast<const GenericConvParams*>(cstmStruct)
            : mlss::conv::utils::buildConvParams(attr);

        bool isArchSupported = isGfx110x(gfxip) || isGfx120x(gfxip);
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
            isSupported = (params.r <= 0x03u)
                       && (params.s <= 0x03u)
                       && (params.groups == 0x01u);
        }

        if (isSupported)
        {
            auto numCuResult = getNumCu(gfxip);
            if (numCuResult.has_value())
            {
                const auto numCu = static_cast<std::uint32_t>(numCuResult.value());
                const std::uint64_t kGroups = integer_divide_ceil(static_cast<std::uint64_t>(params.k),
                                                                  std::uint64_t{0x10u});

                isSupported = (numCu < 0x0100u) && (kGroups <= numCu);
            }
            else
            {
                isSupported = false;
            }
        }

        bool isFullSupport = isSupported;
        if (isFullSupport)
        {
            if ((params.h < 0x20u) && (params.w < 0x20u))
            {
                isFullSupport = false;
            }
        }

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported    ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isFullSupport  ? 0x00000001u : 0x00000000u;
        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        const auto params = cstmStruct
            ? *static_cast<const GenericConvParams*>(cstmStruct)
            : mlss::conv::utils::buildConvParams(attr);

        auto capsResult = isShadersAvailable(gfxip, attr, cstmStruct);
        if (capsResult.support == 0x00000000u)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
        }

        auto numCuResult = getNumCu(gfxip);
        if (!numCuResult.has_value())
        {
            return std::unexpected(numCuResult.error());
        }

        const auto numCu = static_cast<std::uint32_t>(numCuResult.value());
        const auto shader = selectShaderVariant(gfxip, params, numCu);

        const auto shaderPair = selectShaderPair(shader);
        if (shaderPair.reloc.m_binary.empty() || shaderPair.nonReloc.m_binary.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        auto workgroupSizeResult = getWorkgroupSize(shaderPair.reloc.m_binary);
        MLSSdim3 blocks = workgroupSizeResult.value_or(MLSSdim3{ 0x01u, 0x01u, 0x01u });
        MLSSdim3 grid{ numCu, 0x01u, 0x01u };

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

} // namespace mlss::conv::mxn::winograd::fury
