/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::one_by_one::hip::wmma
{

    namespace
    {
        enum class HipConv1x1Shader : std::uint32_t
        {
            Shader16x16x16WMMA_NN = 0x00u,
            Shader16x128x16WMMA_NN,
            Shader16x256x16WMMA_NN,
            Shader64x128x32WMMA_NN,
            Shader128x128x16WMMA_NN,
            Shader128x64x32WMMA_NN,
            Shader256x16x16WMMA_NN,

            Shader64x128x32WMMA_NT,
            Shader64x256x32WMMA_NN,
            Shader64x256x32WMMA_NT,

            ShaderCount
        };

        constexpr auto DefaultShader = HipConv1x1Shader::Shader16x16x16WMMA_NN;

        std::uint64_t makeCacheKey(const GfxIpTriple& gfxip, HipConv1x1Shader shader)
        {
            return (static_cast<std::uint64_t>(gfxIpPacked(gfxip)) << 0x20)
                 | static_cast<std::uint64_t>(shader);
        }

        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        ShaderDescriptorType selectRelocatableShader(const GfxIpTriple& gfxip, HipConv1x1Shader shader)
        {
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x05u)
            {
                switch (shader)
                {
                    case HipConv1x1Shader::Shader16x16x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_NN_16x16x16_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader16x128x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_NN_16x128x16_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader16x256x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_NN_16x256x16_1_2_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader64x128x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_64x128x32_NN_Elf);
                    case HipConv1x1Shader::Shader64x128x32WMMA_NT:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_64x128x32_NT_Elf);
                    case HipConv1x1Shader::Shader64x256x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_NN_64x256x32_2_2_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader64x256x32WMMA_NT:
                        return make_shader_descriptor(fp16::gfx1150::Gemm2d_NT_64x256x32_2_2_RocWmma_F16_Elf);
                    default:
                        break;
                }
            }
            else if (gfxip.major == 0x0Bu)
            {
                switch (shader)
                {
                    case HipConv1x1Shader::Shader16x16x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1100::Gemm2d_NN_16x16x16_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader16x128x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1100::Gemm2d_NN_16x128x16_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader16x256x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1100::Gemm2d_NN_16x256x16_1_2_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader64x128x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1100::Gemm2d_64x128x32_NN_Elf);
                    case HipConv1x1Shader::Shader64x128x32WMMA_NT:
                        return make_shader_descriptor(fp16::gfx1100::Gemm2d_64x128x32_NT_Elf);
                    default:
                        break;
                }
            }
            else if (gfxip.major == 0x0Cu)
            {
                switch (shader)
                {
                    case HipConv1x1Shader::Shader16x16x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::Gemm2d_NN_16x16x16_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader16x128x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::Gemm2d_NN_16x128x16_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader16x256x16WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::Gemm2d_NN_16x256x16_1_2_RocWmma_F16_Elf);
                    case HipConv1x1Shader::Shader64x128x32WMMA_NN:
                        return make_shader_descriptor(fp16::gfx1201::Gemm2d_64x128x32_NN_Elf);
                    case HipConv1x1Shader::Shader64x128x32WMMA_NT:
                        return make_shader_descriptor(fp16::gfx1201::Gemm2d_64x128x32_NT_Elf);
                    default:
                        break;
                }
            }
            return {};
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
            HipConv1x1Shader shader)
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

        std::span<const int> selectConstants(HipConv1x1Shader shader)
        {
            switch (shader)
            {
                case HipConv1x1Shader::Shader16x16x16WMMA_NN:   return fp16::gemm2d_16x16x16NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader16x128x16WMMA_NN:  return fp16::gemm2d_16x128x16NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader16x256x16WMMA_NN:  return fp16::gemm2d_16x256x16NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader64x128x32WMMA_NN:  return fp16::gemm2d_64x128x32NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader128x128x16WMMA_NN: return fp16::gemm2d_128x128x16NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader128x64x32WMMA_NN:  return fp16::gemm2d_128x64x32NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader256x16x16WMMA_NN:  return fp16::gemm2d_256x16x16NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader64x128x32WMMA_NT:  return fp16::gemm2d_64x128x32NT_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader64x256x32WMMA_NN:  return fp16::gemm2d_64x256x32NN_F16_F16_CONSTANTS;
                case HipConv1x1Shader::Shader64x256x32WMMA_NT:  return fp16::gemm2d_64x256x32NN_F16_F16_CONSTANTS;
                default:                                         return fp16::gemm2d_16x16x16NN_F16_F16_CONSTANTS;
            }
        }

        HipConv1x1Shader selectShaderVariant(const GfxIpTriple& gfxip, const GenericConvParams& params)
        {
            const std::array<float, 0x03> features = {
                static_cast<float>(params.k),
                static_cast<float>(params.h * params.w),
                static_cast<float>(params.c)
            };

            auto shader = DefaultShader;

            if (gfxip == IP_GFX1100)
            {
                shader = static_cast<HipConv1x1Shader>(
                    predictHelper(fp16::Navi31HipGemmFp16Tree, std::span{features}));
            }
            else if (gfxip == IP_GFX1101)
            {
                shader = static_cast<HipConv1x1Shader>(
                    predictHelper(fp16::Navi32HipGemmFp16Tree, std::span{features}));
            }
            else if (gfxip == IP_GFX1102 || gfxip == IP_GFX1103)
            {
                shader = static_cast<HipConv1x1Shader>(
                    predictHelper(fp16::Navi33HipGemmFp16Tree, std::span{features}));
            }
            else if (isGfx115x(gfxip))
            {
                shader = static_cast<HipConv1x1Shader>(
                    predictHelper(fp16::StrixHipGemmFp16Tree, std::span{features}));
            }
            else if (gfxip == IP_GFX1201 || gfxip == IP_GFX1200)
            {
                constexpr std::array<std::int32_t, 7> Navi48Labels = {
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader16x16x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader16x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader16x256x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader64x128x32WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader128x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader128x64x32WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader256x16x16WMMA_NN)
                };

                shader = static_cast<HipConv1x1Shader>(
                    predictHelper(fp16::Navi48HipGemmFp16Tree, std::span{features},
                                  std::span{Navi48Labels}));
            }
            else if (gfxip == IP_GFX1210 || gfxip == IP_GFX1211)
            {
                constexpr std::array<std::int32_t, 7> Navi44Labels = {
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader16x16x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader16x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader16x256x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader64x128x32WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader128x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader128x64x32WMMA_NN),
                    static_cast<std::int32_t>(HipConv1x1Shader::Shader256x16x16WMMA_NN)
                };

                shader = static_cast<HipConv1x1Shader>(
                    predictHelper(fp16::Navi44HipGemmFp16Tree, std::span{features},
                                  std::span{Navi44Labels}));
            }

            auto constants = selectConstants(shader);
            auto macroTileM = constants[0x01u];
            auto macroTileN = constants[0x02u];
            auto macroTileK = constants[0x03u];

            if ((static_cast<std::uint32_t>(params.k) < static_cast<std::uint32_t>(macroTileM)) ||
                (params.h * params.w < static_cast<std::uint32_t>(macroTileN)) ||
                (static_cast<std::uint32_t>(params.c) < static_cast<std::uint32_t>(macroTileK)))
            {
                shader = DefaultShader;
            }

            if (selectRelocatableShader(gfxip, shader).m_binary.empty())
            {
                shader = DefaultShader;
            }

            return shader;
        }

        inline constexpr std::array<HyperConsts, 4> HyperSetNavi31 = {HyperConsts{0,3,0,26}, HyperConsts{0,0,0,1}, AlwaysFail, AlwaysFail};
        inline constexpr std::array<HyperConsts, 4> HyperSetNavi32 = {HyperConsts{0,3,0,1}, HyperConsts{1,1,0,41}, AlwaysFail, AlwaysFail};
        inline constexpr std::array<HyperConsts, 4> HyperSetNavi33 = {HyperConsts{0,0,0,1}, HyperConsts{3,0,0,201}, AlwaysFail, AlwaysFail};
        inline constexpr std::array<HyperConsts, 4> HyperSetPhoenix = {HyperConsts{2,3,0,40}, HyperConsts{2,3,28,84}, AlwaysFail, AlwaysFail};

        bool chooseTuning(const GfxIpTriple& gfxip, const GenericConvParams& params)
        {
            const std::size_t mode = static_cast<std::size_t>(params.hasBias)
                                   + 0x02u * static_cast<std::size_t>(params.backward);

            HyperConsts archSet = AlwaysFail;

            if (isGfx110x(gfxip) && gfxip != IP_GFX1103)
            {
                if (gfxip == IP_GFX1100)
                {
                    archSet = HyperSetNavi31[mode];
                }
                else if (gfxip == IP_GFX1101)
                {
                    archSet = HyperSetNavi32[mode];
                }
                else if (gfxip == IP_GFX1102)
                {
                    archSet = HyperSetNavi33[mode];
                }
            }
            else if (isGfx115x(gfxip))
            {
                archSet = HyperSetPhoenix[mode];
            }
            else if (isGfx120x(gfxip))
            {
                return true;
            }

            if (archSet != AlwaysFail)
            {
                return calcHyperboloid(archSet,
                    static_cast<float>(params.c),
                    static_cast<float>(params.k),
                    static_cast<float>(params.n * params.outH * params.outW));
            }

            return false;
        }

    } // namespace

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const GenericConvParams& params)
    {
        using mlss::op::utils::MetaCmdCaps;

        bool isArchSupported = isGfx11(gfxip) || isGfx12(gfxip);

        if (!isArchSupported)
        {
            return MetaCmdCaps{.values = 0x00000000u};
        }

        bool isFp16 = params.dataType == DataTypeFlags::FLOAT16;
        bool isSupported = true;

        if ((params.s != 0x01u) || (params.r != 0x01u))
        {
            isSupported = false;
        }
        else if ((params.outPadX > 0x00u) || (params.outPadY > 0x00u))
        {
            isSupported = false;
        }
        else if ((params.startPadX != 0x00u)   ||
                 (params.startPadY != 0x00u)    ||
                 (params.endPadX != 0x00u)      ||
                 (params.endPadY != 0x00u)      ||
                 (params.outPadX != 0x00u)      ||
                 (params.outPadY != 0x00u)      ||
                 (params.convStrideX != 0x01u)  ||
                 (params.convStrideY != 0x01u)  ||
                 (params.inputStrideX != 0x01u) ||
                 (params.inputStrideY != 0x01u) ||
                 (params.filterStrideX != 0x01u)||
                 (params.filterStrideY != 0x01u)||
                 (params.groups != 0x01u)       ||
                 (params.h * params.w < 0x10u)  ||
                 (params.k < 0x10u)             ||
                 (params.c < 0x10u)             ||
                 (params.backward == true)      ||
                 (!isFp16))
        {
            isSupported = false;
        }
        else if (((params.h * params.w) % 0x02u != 0x00u) ||
                 (params.c % 0x02u != 0x00u) ||
                 (params.k % 0x02u != 0x00u))
        {
            isSupported = false;
        }
        else if (params.dataType == DataTypeFlags::UINT32)
        {
            isSupported = false;
        }
        else if (isFp16 && (params.precision == PrecisionFlags::FLOAT32))
        {
            isSupported = false;
        }
        else if ((params.activation != ActivationFunctionFlags::COUNT) &&
                 ((params.activation == ActivationFunctionFlags::HARDMAX)     ||
                  (params.activation == ActivationFunctionFlags::LOG_SOFTMAX) ||
                  (params.activation == ActivationFunctionFlags::SOFTMAX)))
        {
            isSupported = false;
        }

        bool isFullySupported = isSupported && chooseTuning(gfxip, params);

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported      ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isFullySupported ? 0x00000001u : 0x00000000u;
        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const GenericConvParams& params)
    {
        auto capsResult = isShadersAvailable(gfxip, params);
        if (capsResult.support == 0x00000000u)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
        }

        auto shader = selectShaderVariant(gfxip, params);

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
        auto macroTileM = static_cast<std::uint32_t>(constants[0x01u]);
        auto macroTileN = static_cast<std::uint32_t>(constants[0x02u]);

        std::uint32_t blockCountX = integer_divide_ceil(params.k, macroTileM);
        std::uint32_t blockCountY = integer_divide_ceil(params.h * params.w, macroTileN);

        if (isGfx12(gfxip))
        {
            constexpr std::uint32_t maxBlocks = 0xFFFFu;
            blockCountX = std::min(blockCountX, maxBlocks);
            blockCountY = std::min(blockCountY, maxBlocks);
        }

        MLSSdim3 grid{ blockCountX, blockCountY, params.n };
        MLSSdim3 blocks{
            static_cast<std::uint32_t>(constants[0x04u]),
            static_cast<std::uint32_t>(constants[0x05u]),
            static_cast<std::uint32_t>(constants[0x06u])
        };

        Binaries binaries;

        Blob relocBlob = std::move(*make_binary_blob(relocDescriptor));
        relocBlob = fp16::hip_conv_1x1_ARGS;
        relocBlob.m_constants.assign(constants.begin(), constants.end());
        relocBlob.setGridBlocks(grid, blocks);

        Blob nonRelocBlob = std::move(*make_binary_blob(nonRelocDescriptor));
        nonRelocBlob = fp16::hip_conv_1x1_ARGS;
        nonRelocBlob.m_constants.assign(constants.begin(), constants.end());
        nonRelocBlob.setGridBlocks(grid, blocks);

        binaries.addBlob(std::move(relocBlob));
        binaries.addBlob(std::move(nonRelocBlob));
        return binaries;
    }

} // namespace mlss::conv::one_by_one::hip::wmma
