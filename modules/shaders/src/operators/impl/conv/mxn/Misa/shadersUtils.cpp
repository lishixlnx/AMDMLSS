/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::mxn::misa
{

    namespace
    {
        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        ShaderDescriptorType selectRelocatableShader(const GfxIpTriple& gfxip, const GenericConvParams& params)
        {
            bool hasRelu = params.activation == ActivationFunctionFlags::RELU;
            bool hasBias = params.hasBias;

            if (gfxip.major == 0x0Bu)
            {
                if (hasBias && hasRelu)  return make_shader_descriptor(fp16::gfx1100::MisaConv_Bias_Relu_gfx11);
                if (hasRelu)             return make_shader_descriptor(fp16::gfx1100::MisaConv_Relu_gfx11);
                if (hasBias)             return make_shader_descriptor(fp16::gfx1100::MisaConv_Bias_gfx11);
                return make_shader_descriptor(fp16::gfx1100::MisaConv_gfx11);
            }
            else if (gfxip.major == 0x0Cu)
            {
                if (hasBias && hasRelu)  return make_shader_descriptor(fp16::gfx1201::MisaConv_Bias_Relu_gfx12);
                if (hasRelu)             return make_shader_descriptor(fp16::gfx1201::MisaConv_Relu_gfx12);
                if (hasBias)             return make_shader_descriptor(fp16::gfx1201::MisaConv_Bias_gfx12);
                return make_shader_descriptor(fp16::gfx1201::MisaConv_gfx12);
            }
            return {};
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Cu) return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        std::expected<const DynamicShaderType*, std::error_code> getOrComputeCached(
            const GfxIpTriple& gfxip,
            const GenericConvParams& params)
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

            if (params.hasBias && hasRelu) return fp16::MisaMxNBiasReluConsts;
            if (hasRelu)                   return fp16::MisaMxNReluConsts;
            if (params.hasBias)            return fp16::MisaMxNBiasConsts;
            return fp16::MisaMxNConsts;
        }

    } // namespace

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        using mlss::op::utils::MetaCmdCaps;

        bool isArchSupported = isGfx110x(gfxip);

        bool isFp16 = params.dataType == DataTypeFlags::FLOAT16;
        bool isSupported = isArchSupported && isFp16;

        bool isFullySupported = false;

        if(isSupported)
        {
            auto a = params.precision;
            if(params.precision != PrecisionFlags::FLOAT16)
            {
                isSupported = false;
            }

            if(params.activation != ActivationFunctionFlags::RELU)
            {
                isSupported = false;
            }

            if(((params.c / params.groups) % fp16::VectorC != 0) || ((params.k / params.groups) % fp16::VectorC != 0))
            {
                isSupported = false;
            }

            if(params.backward)
            {
                isSupported = false;
            }

            if((params.s == 1) && (params.r == 1))
            {
                isSupported = false;
            }

            if((params.dCStride != 1) || (params.oKStride != 1) || (params.fCStride != 1))
            {
                isSupported = false;
            }

            const std::uint64_t maxUint32 = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
            const std::uint64_t memorySizeInput = static_cast<std::uint64_t>(params.c) * params.h * params.w * sizeof(std::uint16_t);
            const std::uint64_t memorySizeOutput = static_cast<std::uint64_t>(params.k) * params.outH * params.outW * sizeof(std::uint16_t);

            if((memorySizeInput > maxUint32) || (memorySizeOutput > maxUint32))
            {
                isSupported = false;
            }
        }

        if(isSupported)
        {
            if((params.s > 1) || (params.r > 1))
            {
                if(params.hasBias && ((params.activation == ActivationFunctionFlags::RELU) || (params.activation == ActivationFunctionFlags::COUNT)))
                {
                    isSupported = true;
                }
                else if(!params.hasBias && (params.activation == ActivationFunctionFlags::RELU) )
                {
                    isSupported = true;
                }
                else
                {
                    isSupported = false;
                }
            }
            else
            {
                isSupported = false;
            }
        }

        isFullySupported = isSupported && isGfx110x(gfxip) && (params.crossCorrelation == true) && (params.s != 1);

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported ;
        caps.fullSupport = isFullySupported;
        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        auto capsResult = isShadersAvailable(gfxip, params);
        if (!capsResult.support)
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

} // namespace mlss::conv::mxn::misa
