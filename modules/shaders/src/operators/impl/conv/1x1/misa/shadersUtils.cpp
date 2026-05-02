/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::one_by_one::misa
{

    namespace
    {
        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        ShaderDescriptorType selectRelocatableShader(const GfxIpTriple& gfxip, const GenericConvParams& params)
        {
            bool hasRelu = params.activation == ActivationFunctionFlags::RELU;

            if (gfxip.major == 0x0Bu)
            {
                return params.hasBias ?
                 hasRelu ?
                  make_shader_descriptor(fp16::gfx1100::MisaConv1x1_Bias_Relu_Elf) :
                    make_shader_descriptor(fp16::gfx1100::MisaConv1x1_Bias_Elf) : 
                    hasRelu ?
                     make_shader_descriptor(fp16::gfx1100::MisaConv1x1_Relu_Elf) :
                     make_shader_descriptor(fp16::gfx1100::MisaConv1x1_Elf);
            }
            else if (gfxip.major == 0x0Cu)
            {
                return params.hasBias ?
                 hasRelu ?
                  make_shader_descriptor(fp16::gfx1201::MisaConv1x1_Bias_Relu_Elf) :
                    make_shader_descriptor(fp16::gfx1201::MisaConv1x1_Bias_Elf) : 
                    hasRelu ?
                     make_shader_descriptor(fp16::gfx1201::MisaConv1x1_Relu_Elf) :
                     make_shader_descriptor(fp16::gfx1201::MisaConv1x1_Elf);
            }
            return {};
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Cu) return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        std::expected<const DynamicShaderType*, std::error_code> getOrComputeCached(const GfxIpTriple& gfxip, const GenericConvParams& params)
        {
            auto key = static_cast<std::uint64_t>(gfxIpPacked(gfxip));

            {
                std::lock_guard lock(s_cacheMutex);
                auto it = s_shaderCache.find(key);
                if (it != s_shaderCache.end())
                {
                    return &it->second;
                }
            }

            auto relocDescriptor = selectRelocatableShader(gfxip, params);
            if (relocDescriptor.m_binary.empty())
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
            }

            auto sourceArch = sourceArchForTarget(gfxip);
            auto nonRelocResult = getNonRelocatable(relocDescriptor.m_binary, sourceArch, gfxip);
            if (!nonRelocResult.has_value())
            {
                return std::unexpected(nonRelocResult.error());
            }

            DynamicShaderType cached;
            cached.m_binary.assign(nonRelocResult->begin(), nonRelocResult->end());
            cached.m_kernelName = relocDescriptor.m_kernelName;
            cached.m_compilerVersion = relocDescriptor.m_compilerVersion;
            cached.m_codeObjectVersion = relocDescriptor.m_codeObjectVersion;
            cached.m_isRelocatable = false;
            cached.m_shaderType = relocDescriptor.m_shaderType;

            std::lock_guard lock(s_cacheMutex);
            auto [it, inserted] = s_shaderCache.emplace(key, std::move(cached));
            return &it->second;
        }

        std::span<const std::uint32_t> selectConstants(const GenericConvParams& params)
        {
            bool hasRelu = params.activation == ActivationFunctionFlags::RELU;

            if (params.hasBias && hasRelu) return fp16::Misa1x1BiasReluConsts;
            if (hasRelu)                   return fp16::Misa1x1ReluConsts;
            if (params.hasBias)            return fp16::Misa1x1BiasConsts;
            return fp16::Misa1x1Consts;
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

        bool hasCorrectActivationCondition = (params.activation == ActivationFunctionFlags::RELU) ||
         (params.activation == ActivationFunctionFlags::COUNT);

         bool isSupported = isArchSupported && isFp16 && hasCorrectActivationCondition;
         bool isFullySupported = false;

         if(isSupported)
         {
            if(params.backward)
            {
                isSupported = false;
            }

            if((params.convStrideX != 1) || (params.convStrideY != 1))
            {
                isSupported = false;
            }

            if((params.startPadX != 0) || (params.startPadY != 0))
            {
                isSupported = false;
            }

            if((params.endPadX >= params.w) || (params.endPadY >= params.h))
            {
                isSupported = false;
            }
         }

         if(isSupported)
         {
            isSupported = false;

            union PackedConvShape
            {
                std::array<std::uint16_t, 4> values;
                std::uint64_t packed;
            };

            PackedConvShape current = { .values = {
                static_cast<std::uint16_t>(params.c),
                static_cast<std::uint16_t>(params.k),
                static_cast<std::uint16_t>(params.h),
                static_cast<std::uint16_t>(params.w)
            }};

            auto matchesAny = [&current](const auto& cases)
            {
                for (const auto& entry : cases)
                {
                    if (current.packed == entry.packed)
                    {
                        return true;
                    }
                }
                return false;
            };

            if (isGfx110x(gfxip))
            {
                if (params.hasBias)
                {
                    constexpr std::array<PackedConvShape, 4> reluCases = {{
                        { .values = { 0x0100u, 0x0040u, 0x0038u, 0x0038u } },
                        { .values = { 0x0200u, 0x0080u, 0x001Cu, 0x001Cu } },
                        { .values = { 0x0400u, 0x0100u, 0x000Eu, 0x000Eu } },
                        { .values = { 0x0800u, 0x0200u, 0x0007u, 0x0007u } },
                    }};

                    constexpr std::array<PackedConvShape, 4> nonReluCases = {{
                        { .values = { 0x0040u, 0x0100u, 0x0038u, 0x0038u } },
                        { .values = { 0x0080u, 0x0200u, 0x001Cu, 0x001Cu } },
                        { .values = { 0x0100u, 0x0400u, 0x000Eu, 0x000Eu } },
                        { .values = { 0x0200u, 0x0800u, 0x0007u, 0x0007u } },
                    }};

                    const auto& cases = (params.activation == ActivationFunctionFlags::RELU) ? reluCases : nonReluCases;
                    isSupported = matchesAny(cases);
                }
            }
            else if (isGfx120x(gfxip))
            {
                if (params.hasBias)
                {
                    // resnet50 RELU cases
                    constexpr std::array<PackedConvShape, 5> resnetReluCases = {{
                        { .values = { 0x0040u, 0x0040u, 0x0038u, 0x0038u } },
                        { .values = { 0x0100u, 0x0040u, 0x0038u, 0x0038u } },
                        { .values = { 0x0200u, 0x0080u, 0x001Cu, 0x001Cu } },
                        { .values = { 0x0400u, 0x0100u, 0x000Eu, 0x000Eu } },
                        { .values = { 0x0800u, 0x0200u, 0x0007u, 0x0007u } },
                    }};

                    // resnet50 non-RELU cases
                    constexpr std::array<PackedConvShape, 4> resnetNonReluCases = {{
                        { .values = { 0x0040u, 0x0100u, 0x0038u, 0x0038u } },
                        { .values = { 0x0080u, 0x0200u, 0x001Cu, 0x001Cu } },
                        { .values = { 0x0100u, 0x0400u, 0x000Eu, 0x000Eu } },
                        { .values = { 0x0200u, 0x0800u, 0x0007u, 0x0007u } },
                    }};

                    // inceptionv4 RELU cases
                    constexpr std::array<PackedConvShape, 9> inceptionReluCases = {{
                        { .values = { 0x00A0u, 0x0040u, 0x0049u, 0x0049u } },
                        { .values = { 0x0180u, 0x0060u, 0x0023u, 0x0023u } },
                        { .values = { 0x0180u, 0x0040u, 0x0023u, 0x0023u } },
                        { .values = { 0x0400u, 0x0180u, 0x0011u, 0x0011u } },
                        { .values = { 0x0400u, 0x00C0u, 0x0011u, 0x0011u } },
                        { .values = { 0x0400u, 0x0080u, 0x0011u, 0x0011u } },
                        { .values = { 0x0400u, 0x0100u, 0x0011u, 0x0011u } },
                        { .values = { 0x0600u, 0x0100u, 0x0008u, 0x0008u } },
                        { .values = { 0x0600u, 0x0180u, 0x0008u, 0x0008u } },
                    }};

                    // lightroom RELU cases
                    constexpr std::array<PackedConvShape, 6> lightroomReluCases = {{
                        { .values = { 0x0080u, 0x0080u, 0x0100u, 0x0100u } },
                        { .values = { 0x00C0u, 0x0080u, 0x0100u, 0x0100u } },
                        { .values = { 0x0100u, 0x0080u, 0x0100u, 0x0100u } },
                        { .values = { 0x0080u, 0x0080u, 0x00FFu, 0x00FFu } },
                        { .values = { 0x00C0u, 0x0080u, 0x00FFu, 0x00FFu } },
                        { .values = { 0x0100u, 0x0080u, 0x00FFu, 0x00FFu } },
                    }};

                    // deeplabv3 RELU cases
                    constexpr std::array<PackedConvShape, 2> deeplabReluCases = {{
                        { .values = { 0x0140u, 0x0100u, 0x0041u, 0x0041u } },
                        { .values = { 0x0200u, 0x0100u, 0x0041u, 0x0041u } },
                    }};

                    if (params.activation == ActivationFunctionFlags::RELU)
                    {
                        isSupported = matchesAny(resnetReluCases)
                                   || matchesAny(inceptionReluCases)
                                   || matchesAny(lightroomReluCases)
                                   || matchesAny(deeplabReluCases);
                    }
                    else
                    {
                        isSupported = matchesAny(resnetNonReluCases);
                    }

                    // Cases requiring hasBias but any activation
                    if (!isSupported)
                    {
                        // yolov3
                        constexpr std::array<PackedConvShape, 1> yolov3Cases = {{
                            { .values = { 0x0400u, 0x0200u, 0x000Du, 0x000Du } },
                        }};

                        // lightroom (any activation)
                        constexpr std::array<PackedConvShape, 16> lightroomAnyCases = {{
                            { .values = { 0x0040u, 0x000Cu, 0x00E2u, 0x00E2u } },
                            { .values = { 0x0040u, 0x0003u, 0x01C2u, 0x01C2u } },
                            { .values = { 0x0020u, 0x0003u, 0x01D6u, 0x01D6u } },
                            { .values = { 0x0080u, 0x0080u, 0x0200u, 0x0200u } },
                            { .values = { 0x00C0u, 0x00C0u, 0x0100u, 0x0100u } },
                            { .values = { 0x0100u, 0x0100u, 0x0080u, 0x0080u } },
                            { .values = { 0x0200u, 0x0200u, 0x0040u, 0x0040u } },
                            { .values = { 0x0200u, 0x0200u, 0x0020u, 0x0020u } },
                            { .values = { 0x0200u, 0x0100u, 0x0080u, 0x0080u } },
                            { .values = { 0x0100u, 0x00C0u, 0x0100u, 0x0100u } },
                            { .values = { 0x00C0u, 0x0080u, 0x0200u, 0x0200u } },
                            { .values = { 0x0080u, 0x0003u, 0x0200u, 0x0200u } },
                            { .values = { 0x0040u, 0x0040u, 0x0200u, 0x0200u } },
                            { .values = { 0x0080u, 0x0080u, 0x0100u, 0x0100u } },
                            { .values = { 0x0100u, 0x0080u, 0x0100u, 0x0100u } },
                            { .values = { 0x0080u, 0x0040u, 0x0200u, 0x0200u } },
                        }};

                        // deeplabv3 (any activation)
                        constexpr std::array<PackedConvShape, 18> deeplabAnyCases = {{
                            { .values = { 0x0020u, 0x0010u, 0x0101u, 0x0101u } },
                            { .values = { 0x0010u, 0x0060u, 0x0101u, 0x0101u } },
                            { .values = { 0x0060u, 0x0018u, 0x0081u, 0x0081u } },
                            { .values = { 0x0018u, 0x0090u, 0x0081u, 0x0081u } },
                            { .values = { 0x0090u, 0x0018u, 0x0081u, 0x0081u } },
                            { .values = { 0x0090u, 0x0020u, 0x0041u, 0x0041u } },
                            { .values = { 0x0020u, 0x00C0u, 0x0041u, 0x0041u } },
                            { .values = { 0x00C0u, 0x0020u, 0x0041u, 0x0041u } },
                            { .values = { 0x00C0u, 0x0040u, 0x0041u, 0x0041u } },
                            { .values = { 0x0040u, 0x0180u, 0x0041u, 0x0041u } },
                            { .values = { 0x0180u, 0x0040u, 0x0041u, 0x0041u } },
                            { .values = { 0x0180u, 0x0060u, 0x0041u, 0x0041u } },
                            { .values = { 0x0060u, 0x0240u, 0x0041u, 0x0041u } },
                            { .values = { 0x0240u, 0x0060u, 0x0041u, 0x0041u } },
                            { .values = { 0x0240u, 0x00A0u, 0x0041u, 0x0041u } },
                            { .values = { 0x00A0u, 0x03C0u, 0x0041u, 0x0041u } },
                            { .values = { 0x03C0u, 0x00A0u, 0x0041u, 0x0041u } },
                            { .values = { 0x03C0u, 0x0140u, 0x0041u, 0x0041u } },
                        }};

                        isSupported = matchesAny(yolov3Cases)
                                   || matchesAny(lightroomAnyCases)
                                   || matchesAny(deeplabAnyCases);
                    }
                }
            }
         }

         if(isSupported)
         {
            constexpr std::uint64_t maxUint32 = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());

            const std::uint64_t memorySizeInput = static_cast<std::uint64_t>(params.c) * params.h * params.w * sizeof(std::uint16_t);
            const std::uint64_t memorySizeOutput = static_cast<std::uint64_t>(params.k) * params.outH * params.outW * sizeof(std::uint16_t);

            if((memorySizeInput > maxUint32) || (memorySizeOutput > maxUint32))
            {
                isSupported = false;
            }
         }

         if(isSupported)
         {
            // check if format input is NHWC and output is NHWC
            if ((params.dCStride != 1) && (params.oKStride != 1))
            {
                isSupported = false;
            }
         }

         isFullySupported = isSupported && ((params.filterStrideX == 1) || (params.filterStrideY == 1));

         MetaCmdCaps ret;

         ret.support     = isSupported     ? 0x00000001u : 0x00000000u;
         ret.fullSupport = isFullySupported ? 0x00000001u : 0x00000000u;
         return ret;        
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

        auto relocDescriptor = selectRelocatableShader(gfxip, params);
        if (relocDescriptor.m_binary.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        auto cachedResult = getOrComputeCached(gfxip, params);
        if (!cachedResult.has_value())
        {
            return std::unexpected(cachedResult.error());
        }

        const auto& cachedShader = *cachedResult.value();
        auto nonRelocDescriptor = make_shader_descriptor(cachedShader);

        auto constants = selectConstants(params);
        std::array<std::uint32_t, 3> macroTile = { constants[0], constants[1], constants[2] };
        auto misaArgs = mlss::conv::utils::buildMisaConvArgs(params, constants[2]);
        auto grid = mlss::conv::utils::MisaConvGetGridSize(misaArgs, macroTile);
        MLSSdim3 blocks{ fp16::VectorC * macroTile[0] / macroTile[2], 0x01u, 0x01u };

        Binaries binaries;

        Blob relocBlob = std::move(*make_binary_blob(relocDescriptor));
        relocBlob = fp16::misa_conv_ARGS_CONSTANTS;
        relocBlob.m_constants.assign(constants.begin(), constants.end());
        relocBlob.setGridBlocks(grid, blocks);
        binaries.addBlob(std::move(relocBlob));

        Blob nonRelocBlob = std::move(*make_binary_blob(nonRelocDescriptor));
        nonRelocBlob = fp16::misa_conv_ARGS_CONSTANTS;
        nonRelocBlob.m_constants.assign(constants.begin(), constants.end());
        nonRelocBlob.setGridBlocks(grid, blocks);
        binaries.addBlob(std::move(nonRelocBlob));

        return binaries;
    }

} // namespace mlss::conv::one_by_one::misa
