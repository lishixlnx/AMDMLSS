/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

namespace mlss::conv::dilated::hip::wmma
{

    namespace
    {
        enum class HipConvDilatedShader : std::uint32_t
        {
            Shader32x32x64WMMA_NN = 0,
            Shader64x32x32WMMA_NN,
            Shader64x64x64WMMA_NN,
            Shader128x64x32WMMA_NN,
            Shader256x32x32WMMA_NN,
            Shader128x128x32WMMA_NN,
            ShaderCount
        };

        std::uint64_t makeCacheKey(const GfxIpTriple& gfxip, HipConvDilatedShader shader)
        {
            return (static_cast<std::uint64_t>(gfxIpPacked(gfxip)) << 32)
                 | static_cast<std::uint64_t>(shader);
        }

        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        HipConvDilatedShader retrieveNNShader(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
        {
            auto shader = HipConvDilatedShader::Shader64x64x64WMMA_NN;

            if ((params.s != 0x00000003u) || (params.filterStrideX <= 0x00000001u))
            {
                return shader;
            }

            switch (gfxip.major)
            {
            case 0x0Bu:
                break;
            case 0x0Cu:
                if ((((params.c < 0x00000100u) && (params.k < 0x00000100u)) && ((params.h < 0x00000040u) || (params.w < 0x00000080u))) ||
                    (((params.c <= 0x00000100u) && (params.k <= 0x00000100u)) && (params.h == params.w)))
                {
                    shader = HipConvDilatedShader::Shader32x32x64WMMA_NN;
                }
                else if (((params.c < 0x00000040u) && (params.k < 0x00000040u)) &&
                         ((params.h >= 0x00000040u) || (params.w >= 0x00000080u)) &&
                         ((params.h < 0x00000100u) || (params.w < 0x00000140u)))
                {
                    shader = HipConvDilatedShader::Shader64x32x32WMMA_NN;
                }
                else if (((params.c < 0x00000040u) && (params.k < 0x00000040u)) &&
                         ((params.h >= 0x00000100u) || (params.w >= 0x00000140u)))
                {
                    shader = HipConvDilatedShader::Shader256x32x32WMMA_NN;
                }
                else if (((params.c < 0x00000400u) && (params.k < 0x00000400u)) &&
                         ((params.c >= 0x00000100u) || (params.k >= 0x00000100u)) &&
                         ((params.h <= 0x00000020u) && (params.w <= 0x00000040u)))
                {
                    shader = HipConvDilatedShader::Shader64x64x64WMMA_NN;
                }
                else if ((((params.c < 0x00000800u) && (params.k < 0x00000800u)) && ((params.c >= 0x00000400u) && (params.k >= 0x00000400u)) &&
                          ((params.h <= 0x00000020u) || (params.w <= 0x00000040u))) ||
                         (((params.c < 0x00000100u) && (params.k < 0x00000100u)) && ((params.c >= 0x00000040u) || (params.k >= 0x00000040u)) &&
                          ((params.h >= 0x00000040u) || (params.w >= 0x00000080u))))
                {
                    shader = HipConvDilatedShader::Shader128x64x32WMMA_NN;
                }
                else if ((((params.c >= 0x00000800u) && (params.k >= 0x00000800u)) && ((params.h >= 0x00000010u) || (params.w >= 0x00000020u))) ||
                         (((params.c >= 0x00000100u) && (params.k >= 0x00000100u)) && ((params.h >= 0x00000020u) || (params.w >= 0x00000040u))))
                {
                    shader = HipConvDilatedShader::Shader128x128x32WMMA_NN;
                }
                break;
            default:
                break;
            }
            return shader;
        }

        ShaderDescriptorType selectRelocatableShader(const GfxIpTriple& gfxip, HipConvDilatedShader shader)
        {
            if (gfxip.major == 0x0Bu)
            {
                if (gfxip.minor == 0x00u)
                {
                    return make_shader_descriptor(fp16::gfx1100::grouped_conv_fwd_bias_relu_add_wmma_fp16_hip_amdgcn_amd_amdhsa_gfx1100_coba);
                }
                if (gfxip.minor == 0x05u)
                {
                    return make_shader_descriptor(fp16::gfx1150::grouped_conv_fwd_bias_relu_add_wmma_fp16_hip_amdgcn_amd_amdhsa_gfx1150_coba);
                }
            }
            else if (gfxip.major == 0x0Cu)
            {
                switch (shader)
                {
                    case HipConvDilatedShader::Shader32x32x64WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::gemm2d_NN_32x32x64_F16);
                    case HipConvDilatedShader::Shader64x32x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::gemm2d_NN_64x32x32NN_F16);
                    case HipConvDilatedShader::Shader64x64x64WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::gemm2d_NN_64x64x64NN_F16);
                    case HipConvDilatedShader::Shader128x64x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::gemm2d_NN_128x64x32NN_F16);
                    case HipConvDilatedShader::Shader256x32x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::gemm2d_NN_256x32x32_F16);
                    case HipConvDilatedShader::Shader128x128x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::gemm2d_NN_128x128x32_F16);
                    default:
                        break;
                }
            }
            return {};
        }

        std::span<const std::uint32_t> selectConstants(HipConvDilatedShader shader)
        {
            switch (shader)
            {
                case HipConvDilatedShader::Shader32x32x64WMMA_NN:  return fp16::gemm2d_32x32x64NN_F16_F16_CONSTANTS;
                case HipConvDilatedShader::Shader64x32x32WMMA_NN:  return fp16::gemm2d_64x32x32NN_F16_F16_CONSTANTS;
                case HipConvDilatedShader::Shader64x64x64WMMA_NN:  return fp16::gemm2d_64x64x64NN_F16_F16_CONSTANTS;
                case HipConvDilatedShader::Shader128x64x32WMMA_NN: return fp16::gemm2d_128x64x32NN_F16_F16_CONSTANTS;
                case HipConvDilatedShader::Shader256x32x32WMMA_NN: return fp16::gemm2d_256x32x32NN_F16_F16_CONSTANTS;
                case HipConvDilatedShader::Shader128x128x32WMMA_NN:return fp16::gemm2d_128x128x32NN_F16_F16_CONSTANTS;
                default: return {};
            }
        }

        std::pair<MLSSdim3, MLSSdim3> calcGridAndBlocks(
            std::span<const std::uint32_t> constants,
            const mlss::conv::utils::GenericConvParams& params)
        {
            auto macroTileM = constants[0x01u];
            auto macroTileN = constants[0x02u];
            auto blockCountX = integer_divide_ceil(params.n * params.outH * params.outW, macroTileM)
                             * integer_divide_ceil(params.k, macroTileN);

            MLSSdim3 grid{ blockCountX, 0x01u, 0x01u };
            MLSSdim3 blocks{ constants[0x04u], constants[0x05u], constants[0x06u] };
            return { grid, blocks };
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x00u) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x05u) return {0x0Bu, 0x05u, 0x00u};
            if (gfxip.major == 0x0Cu)                         return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        std::expected<const DynamicShaderType*, std::error_code> getOrComputeCached(
            const GfxIpTriple& gfxip,
            HipConvDilatedShader shader)
        {
            auto key = makeCacheKey(gfxip, shader);

            {
                std::lock_guard lock(s_cacheMutex);
                auto it = s_shaderCache.find(key);
                if (it != s_shaderCache.end())
                {
                    return &it->second;
                }
            }

            auto relocDescriptor = selectRelocatableShader(gfxip, shader);
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

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        using mlss::op::utils::MetaCmdCaps;

        bool isArchSupported = isGfx110x(gfxip) || isGfx115x(gfxip) || isGfx120x(gfxip);

        if (!isArchSupported)
        {
            return MetaCmdCaps{.values = 0x00000000u};
        }

        bool isFp16 = params.dataType == DataTypeFlags::FLOAT16;
        bool isRelu = params.activation == ActivationFunctionFlags::RELU;

        bool isSupported = isFp16
                        && isRelu
                        && (params.n == 0x00000001u)
                        && (params.groups == 0x00000001u);

        bool isFullySupported = isSupported
                             && (params.w > 0x00000001u)
                             && (params.filterStrideX > 0x00000001u);

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported     ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isFullySupported ? 0x00000001u : 0x00000000u;
        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        auto shader = retrieveNNShader(gfxip, params);

        if ((gfxip.major == 0x0Bu) && (shader != HipConvDilatedShader::Shader64x64x64WMMA_NN))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
        }

        auto relocDescriptor = selectRelocatableShader(gfxip, shader);
        if (relocDescriptor.m_binary.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        auto cachedResult = getOrComputeCached(gfxip, shader);
        if (!cachedResult.has_value())
        {
            return std::unexpected(cachedResult.error());
        }

        const auto& cachedShader = *cachedResult.value();
        auto nonRelocDescriptor = make_shader_descriptor(cachedShader);

        auto constants = selectConstants(shader);
        auto [grid, blocks] = calcGridAndBlocks(constants, params);

        Binaries binaries;

        Blob relocBlob = std::move(*make_binary_blob(relocDescriptor));
        relocBlob = fp16::dilated_convolution_ARGS_CONSTANTS;
        relocBlob.m_constants.assign(constants.begin(), constants.end());
        relocBlob.setGridBlocks(grid, blocks);

        Blob nonRelocBlob = std::move(*make_binary_blob(nonRelocDescriptor));
        nonRelocBlob = fp16::dilated_convolution_ARGS_CONSTANTS;
        nonRelocBlob.m_constants.assign(constants.begin(), constants.end());
        nonRelocBlob.setGridBlocks(grid, blocks);

        binaries.addBlob(std::move(relocBlob));
        binaries.addBlob(std::move(nonRelocBlob));
        return binaries;
    }

} // namespace mlss::conv::dilated::hip::wmma
