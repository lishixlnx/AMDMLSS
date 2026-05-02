/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1100/fp16/shadersBinReloc.hpp"
#include "gfx1100/fp16/shadersBinNonReloc.hpp"
#include "gfx1100/fp32/shadersBinReloc.hpp"
#include "gfx1100/fp32/shadersBinNonReloc.hpp"
#include "gfx1201/fp16/shadersBinReloc.hpp"
#include "gfx1201/fp16/shadersBinNonReloc.hpp"
#include "gfx1201/fp32/shadersBinReloc.hpp"
#include "gfx1201/fp32/shadersBinNonReloc.hpp"

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::mxn::winograd::base
{

    using namespace mlss::conv::mxn::winograd::base::fp16;
    using namespace mlss::conv::mxn::winograd::base::fp32;

    namespace
    {

        enum StrideMode : std::uint32_t
        {
            StrideModeDefault = 0x00u,
            StrideModeDecimation,
            StrideModeDilation,
            StrideModeCount
        };

        // Pairs the relocatable Winograd ELF (used at the cross-compile entry
        // point) with its pre-linked, non-relocatable sibling shipped under
        // shadersBinNonReloc.hpp. The previous design computed the non-reloc
        // half at runtime via comgr's LINK_RELOCATABLE_TO_EXECUTABLE; that
        // path silently exits on PAL OS/ABI inputs, which is what every
        // Winograd ELF is. The shipped _NonReloc binaries are produced offline
        // by tools/winograd_pal/ from the same relocatables.
        struct ShaderPair
        {
            ShaderDescriptorType reloc;
            ShaderDescriptorType nonReloc;
        };

        // Symbol pair shorthand: NS::NAME and NS::NAME_NonReloc live in the
        // same namespace by construction (see tools/winograd_pal/reemit.py).
#define MLSS_WINOGRAD_BASE_PAIR(NS, NAME)                                      \
    ShaderPair                                                                 \
    {                                                                          \
        .reloc    = make_shader_descriptor(NS::NAME),                          \
        .nonReloc = make_shader_descriptor(NS::NAME##_NonReloc),               \
    }

        //=============================================================================================================
        template <class T>
            requires std::is_integral_v<T>
        T roundUpToMultiple(const T& value, const std::type_identity_t<T>& multiple)
        {
            return integer_divide_ceil(value, multiple) * multiple;
        }

        //=============================================================================================================
        float computeGranularityLoss(
            const GenericConvParams& params,
            StrideMode strideMode,
            std::uint32_t numCu,
            bool forceFilterTraverseMode,
            bool filterTraverseModeDual,
            bool isF3x2)
        {
            const std::uint64_t tileR  = isF3x2 ? 0x02u : 0x03u;
            const std::uint64_t tileS  = isF3x2 ? 0x02u : 0x03u;
            const std::uint64_t tileOW = isF3x2 ? 0x03u : 0x02u;
            const std::uint64_t tileOH = isF3x2 ? 0x03u : 0x02u;

            const bool isStride1    = (strideMode == StrideModeDefault);
            const bool isStride2Dec = (strideMode == StrideModeDecimation);
            const bool isStride2Dil = (strideMode == StrideModeDilation);

            const bool isDual = (forceFilterTraverseMode && filterTraverseModeDual)
                             || (!forceFilterTraverseMode && params.s > tileS)
                             || isStride2Dec || isStride2Dil;

            const std::uint64_t rFactor = tileR
                * ((isStride1 || (static_cast<std::uint64_t>(params.r) % (tileR * 0x02u)) == 0x01u) ? 0x01u : 0x02u);
            const std::uint64_t sFactor   = tileS * (isDual ? 0x02u : 0x01u);
            const std::uint64_t cFactor   = (params.dataType == DataTypeFlags::FLOAT32) ? 0x01u : 0x02u;
            const std::uint64_t kFactor   = isStride2Dil ? 0x10u : 0x20u;
            constexpr std::uint64_t nhwFactor = 0x20u;
            const std::uint64_t owFactor  = tileOW * params.inputStrideX;
            const std::uint64_t ohFactor  = tileOH * params.inputStrideY;

            std::uint64_t oW1 = params.outW;
            std::uint64_t oH1 = params.outH;
            if (isStride2Dil && (params.startPadX % 0x02u == 0x00u))
            {
                oW1++;
            }
            if (isStride2Dil && (params.startPadY % 0x02u == 0x01u))
            {
                oH1++;
            }

            const std::uint64_t c = static_cast<std::uint64_t>(params.c) / params.groups;
            const std::uint64_t k = static_cast<std::uint64_t>(params.k) / params.groups;

            const std::uint64_t Rg  = roundUpToMultiple(static_cast<std::uint64_t>(params.r), rFactor);
            const std::uint64_t Sg  = roundUpToMultiple(static_cast<std::uint64_t>(params.s), sFactor);
            const std::uint64_t Cg  = roundUpToMultiple(c, cFactor);
            const std::uint64_t Kg  = roundUpToMultiple(k, kFactor);
            const std::uint64_t oWg = roundUpToMultiple(oW1, owFactor);
            const std::uint64_t oHg = roundUpToMultiple(oH1, ohFactor);

            const std::uint64_t groups   = params.groups;
            const std::uint64_t nGroups  = integer_divide_ceil(static_cast<std::uint64_t>(numCu), groups);
            const std::uint64_t nhwTiles = static_cast<std::uint64_t>(params.n)
                                         * integer_divide_ceil(oHg, ohFactor)
                                         * integer_divide_ceil(oWg, owFactor);

            const std::uint64_t nkhwPerWork = kFactor * nhwFactor * tileOH * tileOW;
            const std::uint64_t nWorks      = integer_divide_ceil(k, kFactor)
                                            * integer_divide_ceil(nhwTiles, nhwFactor);
            const std::uint64_t nWorksPerCu = integer_divide_ceil(nWorks, nGroups)
                                            * integer_divide_ceil(groups * nGroups,
                                                                  static_cast<std::uint64_t>(numCu));

            std::ignore = Kg;

            const std::uint64_t macsg = nWorksPerCu * numCu * nkhwPerWork * Cg * Rg * Sg;
            const std::uint64_t macs  = static_cast<std::uint64_t>(params.n) * groups * k * c
                * integer_divide_ceil(
                    static_cast<std::uint64_t>(params.outH) * params.r,
                    static_cast<std::uint64_t>(params.inputStrideY))
                * integer_divide_ceil(
                    static_cast<std::uint64_t>(params.outW) * params.s,
                    static_cast<std::uint64_t>(params.inputStrideX));

            if (macsg == 0x00u)
            {
                return 1.0f;
            }

            return static_cast<float>(macsg - macs) / static_cast<float>(macsg);
        }

        //=============================================================================================================
        bool selectTileShape(
            const GenericConvParams& params,
            StrideMode strideMode,
            std::uint32_t numCu)
        {
            const bool forceFilterTraverseMode = (strideMode == StrideModeDefault);

            const float lossF23Single = computeGranularityLoss(params, strideMode, numCu, forceFilterTraverseMode, false, false);
            const float lossF23Dual   = computeGranularityLoss(params, strideMode, numCu, forceFilterTraverseMode, true,  false);
            const float lossF32Single = computeGranularityLoss(params, strideMode, numCu, forceFilterTraverseMode, false, true);
            const float lossF32Dual   = computeGranularityLoss(params, strideMode, numCu, forceFilterTraverseMode, true,  true);

            float minLoss = lossF23Single;
            bool isF3x2 = false;

            if (lossF23Dual < minLoss)
            {
                minLoss = lossF23Dual;
                isF3x2 = false;
            }
            if (lossF32Single < minLoss)
            {
                minLoss = lossF32Single;
                isF3x2 = true;
            }
            if (lossF32Dual < minLoss)
            {
                isF3x2 = true;
            }

            return isF3x2;
        }

        //=============================================================================================================
        StrideMode resolveStrideMode(const GenericConvParams& params)
        {
            if ((params.inputStrideX == 0x01u) && (params.convStrideX == 0x02u))
            {
                return StrideModeDecimation;
            }
            if ((params.inputStrideX == 0x02u) && (params.convStrideX == 0x01u))
            {
                return StrideModeDilation;
            }
            return StrideModeDefault;
        }

        //=============================================================================================================
        ShaderPair selectShaderPair(
            const GfxIpTriple& gfxip,
            bool isFp32,
            StrideMode strideMode,
            bool isF3x2)
        {
            if (isGfx110x(gfxip))
            {
                if (isFp32)
                {
                    if (isF3x2)
                    {
                        if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1100, ConvWinogradElf_Gfx11_F3x2_Fp32Stride2Dec);
                        if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1100, ConvWinogradElf_Gfx11_F3x2_Fp32Stride2Dil);
                        return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1100, ConvWinogradElf_Gfx11_F3x2_Fp32Stride1);
                    }
                    if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1100, ConvWinogradElf_Gfx11_F2x3_Fp32Stride2Dec);
                    if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1100, ConvWinogradElf_Gfx11_F2x3_Fp32Stride2Dil);
                    return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1100, ConvWinogradElf_Gfx11_F2x3_Fp32Stride1);
                }

                if (isF3x2)
                {
                    if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1100, ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dec);
                    if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1100, ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dil);
                    return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1100, ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride1);
                }
                if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1100, ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dec);
                if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1100, ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dil);
                return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1100, ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride1);
            }

            if (isGfx120x(gfxip))
            {
                if (isFp32)
                {
                    if (isF3x2)
                    {
                        if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1201, ConvWinogradElf_Gfx12_F3x2_Fp32Stride2Dec);
                        if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1201, ConvWinogradElf_Gfx12_F3x2_Fp32Stride2Dil);
                        return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1201, ConvWinogradElf_Gfx12_F3x2_Fp32Stride1);
                    }
                    if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1201, ConvWinogradElf_Gfx12_F2x3_Fp32Stride2Dec);
                    if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1201, ConvWinogradElf_Gfx12_F2x3_Fp32Stride2Dil);
                    return MLSS_WINOGRAD_BASE_PAIR(fp32::gfx1201, ConvWinogradElf_Gfx12_F2x3_Fp32Stride1);
                }

                if (isF3x2)
                {
                    if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1201, ConvWinogradElf_Gfx12_F3x2_Fp16Dot2Stride2Dec);
                    if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1201, ConvWinogradElf_Gfx12_F3x2_Fp16Dot2Stride2Dil);
                    return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1201, ConvWinogradElf_Gfx12_F3x2_Fp16Dot2Stride1);
                }
                if (strideMode == StrideModeDecimation) return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1201, ConvWinogradElf_Gfx12_F2x3_Fp16Dot2Stride2Dec);
                if (strideMode == StrideModeDilation)   return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1201, ConvWinogradElf_Gfx12_F2x3_Fp16Dot2Stride2Dil);
                return MLSS_WINOGRAD_BASE_PAIR(fp16::gfx1201, ConvWinogradElf_Gfx12_F2x3_Fp16Dot2Stride1);
            }

            return {};
        }

        //=============================================================================================================
        bool insideHyperboloidSet(
            const WinogradHyperSet& hyperSet,
            const GenericConvParams& params,
            bool isStride2,
            bool isFp32)
        {
            const bool isForward = !params.backward;
            const std::uint32_t offset = static_cast<std::uint32_t>(isStride2) * 0x04u
                                       + static_cast<std::uint32_t>(isForward) * 0x02u
                                       + static_cast<std::uint32_t>(isFp32);

            const std::uint32_t rs = std::min(params.r, params.s);
            const auto& vals = (rs <= 0x03u) ? hyperSet.f3x3[offset]
                             : (rs <= 0x05u) ? hyperSet.f5x5[offset]
                             :                 hyperSet.f7x7[offset];

            if (vals.u64All == AlwaysFail.u64All) return false;
            if (vals.u64All == AlwaysPass.u64All) return true;

            const float c = static_cast<float>(params.c);
            const float k = static_cast<float>(params.k);
            const float d = static_cast<float>(params.n)
                          * static_cast<float>(params.outH)
                          * static_cast<float>(params.outW);
            return calcHyperboloid(vals, c, k, d);
        }

        //=============================================================================================================
        std::pair<MLSSdim3, MLSSdim3> calcGridAndBlocks(
            const ShaderDescriptorType& shader,
            std::uint32_t numCu,
            std::uint32_t groups)
        {
            auto workgroupSizeResult = getWorkgroupSize(shader.m_binary);
            MLSSdim3 blocks = workgroupSizeResult.value_or(MLSSdim3{ 0x01u, 0x01u, 0x01u });
            MLSSdim3 grid{ numCu * groups, 0x01u, 0x01u };
            return { grid, blocks };
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
        bool isFp32 = params.dataType == DataTypeFlags::FLOAT32;
        bool isPrecisionFp16 = params.precision == PrecisionFlags::FLOAT16;
        bool isPrecisionFp16Dot2 = params.precision == PrecisionFlags::FLOAT16_ADD_FLOAT32;
        bool isPrecisionFp32 = params.precision == PrecisionFlags::FLOAT32;


        bool isSupported = isArchSupported
                        && (isFp16 || isFp32)
                        && (isPrecisionFp16 || isPrecisionFp16Dot2 || isPrecisionFp32);

        bool isStride2Dil = false;
        bool isStride2Dec = false;

        if (isSupported)
        {
            if ((params.convStrideX   == params.convStrideY)   &&
                (params.inputStrideX  == params.inputStrideY)  &&
                (params.filterStrideX == params.filterStrideY) &&
                (params.filterStrideX == 0x01u))
            {
                bool isStride1 = (params.inputStrideX == 0x01u) && (params.convStrideX == 0x01u);
                isStride2Dec   = (params.inputStrideX == 0x01u) && (params.convStrideX == 0x02u);
                isStride2Dil   = (params.inputStrideX == 0x02u) && (params.convStrideX == 0x01u);

                isSupported = isStride1 || isStride2Dec || isStride2Dil;
            }
            else
            {
                isSupported = false;
            }
        }

        if (isSupported)
        {
            // const std::uint32_t paddedS   = (params.s - 0x01u) * params.filterStrideX + 0x01u;
            // const std::uint32_t paddedR   = (params.r - 0x01u) * params.filterStrideY + 0x01u;
            const std::uint32_t elemSize  = isFp32 ? 0x04u : 0x02u;
            // constexpr std::uint32_t Uint22Max = (0x01u << 22) - 0x01u;
            // constexpr std::uint32_t Uint28Max = (0x01u << 28) - 0x01u;
            constexpr std::uint32_t TwoGBytes = (0x01u << 31);
    
            if ((params.w         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.h         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.c         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.n         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.k         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.r         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.s         > std::numeric_limits<std::uint16_t>::max()) ||
                (params.outW      > std::numeric_limits<std::uint16_t>::max()) ||
                (params.outH      > std::numeric_limits<std::uint16_t>::max()) ||
                (params.startPadX > std::numeric_limits<std::uint16_t>::max()) ||
                (params.startPadY > std::numeric_limits<std::uint16_t>::max()))
            {
                // W, H, C, N, K, R, S, outW, outH, startPadX, and startPadY must be less than 2^16.
                isSupported = false;
            }
            else if (((params.groups > 0x01u) && (params.groups == params.c)) ||
                     ((params.groups > 0x01u) && (params.groups == params.k)) ||
                     ((params.groups > 0x01u) && !isGfx12(gfxip))            ||
                     ((params.groups > 0x01u) && !isFp16))
            {

                isSupported = false;
            }
            else if ((params.outPadX > 0x00u) || (params.outPadY > 0x00u))
            {
                isSupported = false;
            }
            else if (params.activation != ActivationFunctionFlags::COUNT)
            {
                bool isGfx10Activation = (params.activation == ActivationFunctionFlags::IDENTITY)
                                      || (params.activation == ActivationFunctionFlags::RELU)
                                      || (params.activation == ActivationFunctionFlags::LEAKY_RELU);

                bool isGfx11PlusActivation = isGfx10Activation
                                          || (params.activation == ActivationFunctionFlags::SIGMOID)
                                          || (params.activation == ActivationFunctionFlags::TANH)
                                          || (params.activation == ActivationFunctionFlags::SCALED_TANH);

                if (isGfx10(gfxip) && !isGfx10Activation)
                {
                    isSupported = false;
                }
                else if ((isGfx11(gfxip) || isGfx12(gfxip)) && !isGfx11PlusActivation)
                {
                    isSupported = false;
                }
            }
            else
            {
                // In v21_1_2 and newer...
                if (static_cast<std::uint64_t>(params.k) * params.c * params.r * params.s * elemSize > TwoGBytes)
                {
                    isSupported = false;
                }
                else if ((static_cast<std::uint64_t>(params.n) * params.c * params.h    * params.w    * elemSize > TwoGBytes) ||
                         (static_cast<std::uint64_t>(params.n) * params.k * params.outH * params.outW * elemSize > TwoGBytes))
                {
                    std::uint32_t paddedOutH = params.outH;
                    std::uint32_t paddedOutW = params.outW;

                    if (isStride2Dil)
                    {
                        paddedOutH = (params.outH + 0x01u + ((params.startPadY % 0x02u) == 0x01u)) / 0x02u;
                        paddedOutW = (params.outW + 0x01u + ((params.startPadX % 0x02u) == 0x00u)) / 0x02u;
                    }

                    const std::uint32_t numOutTiles = ((paddedOutW + 0x01u) & ~0x01u) * ((paddedOutH + 0x01u) & ~0x01u);

                    if (((numOutTiles % 0x20u) != 0x00u) ||
                        (static_cast<std::uint64_t>(params.c) * params.h    * params.w    * elemSize > TwoGBytes) ||
                        (static_cast<std::uint64_t>(params.k) * params.outH * params.outW * elemSize > TwoGBytes))
                    {
                        isSupported = false;
                    }
                }
            }
        }

        if (isSupported)
        {
            bool isStride2 = isStride2Dec || isStride2Dil;

            bool isSdOverride = isGfx120x(gfxip)
                             && (params.r == 3)   && (params.s == 3)
                             && (params.c == 640)  && (params.n == 2) && (params.k == 640)
                             && (params.w == 32)   && (params.h == 32)
                             && isStride2Dec;

            if (!isSdOverride)
            {
                const auto& hyperSet = (gfxip == IP_GFX1100) ? HyperSetNavi31
                                     : (gfxip == IP_GFX1101) ? HyperSetNavi32
                                     :                         HyperSetNavi33;

                if (!insideHyperboloidSet(hyperSet, params, isStride2, isFp32))
                {
                    isSupported = (params.c >= 0x20u) && (params.k >= 0x0400u);
                }
            }
        }

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isSupported ? 0x00000001u : 0x00000000u;
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

        const bool isFp32         = (params.dataType == DataTypeFlags::FLOAT32);
        const StrideMode strideMode = resolveStrideMode(params);

        auto numCuResult = getNumCu(gfxip);
        if (!numCuResult.has_value())
        {
            return std::unexpected(numCuResult.error());
        }

        const auto numCu = static_cast<std::uint32_t>(numCuResult.value());
        const bool isF3x2 = selectTileShape(params, strideMode, numCu);

        const auto shaderPair = selectShaderPair(gfxip, isFp32, strideMode, isF3x2);
        if (shaderPair.reloc.m_binary.empty() || shaderPair.nonReloc.m_binary.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        auto [grid, blocks] = calcGridAndBlocks(shaderPair.reloc, numCu, params.groups);

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

} // namespace mlss::conv::mxn::winograd::base
