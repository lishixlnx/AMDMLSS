/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "fp16/gfx1100/shadersBin.hpp"
#include "fp16/gfx1201/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

namespace mlss::conv::one_by_one::misa
{

    namespace
    {
        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        ShaderDescriptorType selectRelocatableShader(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu)
            {
                return make_shader_descriptor(fp16::gfx1100::MisaConv1x1_Elf);
            }
            else if (gfxip.major == 0x0Cu)
            {
                return make_shader_descriptor(fp16::gfx1201::MisaConv1x1_Elf);
            }
            return {};
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Cu) return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        std::expected<const DynamicShaderType*, std::error_code> getOrComputeCached(const GfxIpTriple& gfxip)
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

            auto relocDescriptor = selectRelocatableShader(gfxip);
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

    } // namespace

    mlss::op::utils::MetaCmdCaps isMisaShadersAvailable(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        using mlss::op::utils::MetaCmdCaps;

        bool isArchSupported = isGfx110x(gfxip) || isGfx120x(gfxip);

        if (!isArchSupported)
        {
            return MetaCmdCaps{.values = 0x00000000u};
        }

        bool isFp16 = params.dataType == DataTypeFlags::FLOAT16;

        bool isSupported = isFp16
                        && (params.r == 0x00000001u)
                        && (params.s == 0x00000001u);

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isSupported ? 0x00000001u : 0x00000000u;
        return caps;
    }

    std::expected<Binaries, std::error_code> getMisaShadersBlob(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        auto capsResult = isMisaShadersAvailable(gfxip, params);
        if (capsResult.support == 0x00000000u)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
        }

        auto relocDescriptor = selectRelocatableShader(gfxip);
        if (relocDescriptor.m_binary.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        auto cachedResult = getOrComputeCached(gfxip);
        if (!cachedResult.has_value())
        {
            return std::unexpected(cachedResult.error());
        }

        const auto& cachedShader = *cachedResult.value();
        auto nonRelocDescriptor = make_shader_descriptor(cachedShader);

        Binaries binaries;

        Blob relocBlob = std::move(*make_binary_blob(relocDescriptor));
        binaries.addBlob(std::move(relocBlob));

        Blob nonRelocBlob = std::move(*make_binary_blob(nonRelocDescriptor));
        binaries.addBlob(std::move(nonRelocBlob));

        return binaries;
    }

} // namespace mlss::conv::one_by_one::misa
