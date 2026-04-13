/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "gfx1010/fp16/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

using mlss::conv::utils::GenericConvParams;

namespace mlss::conv::mxn::winograd::ultra
{

    namespace
    {
        // These are the maximum supported values for C and K.
        constexpr std::uint32_t MaxC = 240;
        constexpr std::uint32_t MaxK = 16;
        constexpr std::uint16_t Off = std::numeric_limits<std::uint16_t>::max();

        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        ShaderDescriptorType selectRelocatableShader(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Au)
            {
                return make_shader_descriptor(fp16::gfx1010::ConvUltra_Eval_Elf);
            }
            return {};
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Au) return {0x0Au, 0x01u, 0x00u};
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



        bool ComputeCbStrides(const GenericConvParams& params)
        {
            bool supported = false;

            const uint32 tiles_n_row    = RoundUpQuotient(params.outW, o_tile_step_W);
            const uint32 tiles_n_column = RoundUpQuotient(params.outH, o_tile_step_H);

            const uint32 dWPitch     = sizeof(int16);
            const uint64 dHPitch     = dWPitch * params.w;
            const uint64 dCPitch     = dHPitch * params.h;
            const uint64 dNPitch     = dCPitch * params.c;
            const int64  dStep1Pitch = d_tile_step_H * dHPitch - tiles_n_row * d_tile_step_W * dWPitch;
            const int64  dStep2Pitch = dNPitch - tiles_n_column * d_tile_step_H * dHPitch;

            const uint32 oWPitch     = sizeof(int16);
            const uint64 oHPitch     = oWPitch * params.outW;
            const uint64 oKPitch     = oHPitch * params.outH;
            const uint64 oNPitch     = oKPitch * params.k;
            const int64  oStep1Pitch = o_tile_step_H * oHPitch - tiles_n_row * o_tile_step_W * oWPitch;
            const int64  oStep2Pitch = oNPitch - tiles_n_column * o_tile_step_H * oHPitch;

            if ((dCPitch     < (1u << 31)) &&
                (dHPitch     < (1u << 27)) &&
                (oHPitch     < (1u << 27)) &&
                (dStep1Pitch < (1u << 23)) &&
                (oStep1Pitch < (1u << 23)) &&
                (dStep1Pitch >= 0)         &&
                (oStep1Pitch >= 0))
            {
                supported = true;
            }

            return supported;
        }


        struct DLimits
        {
            std::array<std::array<std::uint16_t, 15>, 4> m_roots;
        };

        // For this metacommand, we ran tuning runs over three fixed filter sizes (1x1, 2x2, 3x3) in forward and backward mode.
        // This was seem as a reasonable tradeoff between covering more cases and tuning time.
        struct DLimitsSet
        {
            DLimits f1x1;
            DLimits f2x2;
            DLimits f3x3;
        };

        constexpr DLimitsSet DLimitsSetNavi21 =
        {
            { // f1x1, accuracy: 91.6%
                {
                    { 408, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off },
                    { 288, 280, 248, 312, 304, 504, 504, 504, 504, 504, 504, 504, 504, 504, 504 },
                    { 288, 296, 240, Off, 296, Off, 496, Off, Off, Off, Off, Off, Off, Off, Off },
                    { 240, 248, 184, 184, 144, 144, 144, 160, 120, 160, 104, 160, 112, 120, 112 },
                },
            },
            { // f2x2, accuracy: 89.4%
                {
                    { 144, 32, 24, 16, 16, 16, 40, 40, 40, 48, 40, 48, 40, 40, 40 },
                    { 80,  48, 16, 16, 16, 32, 24, 32, 24, 32, 32, 32, 32, 32, 32 },
                    { 96,  48, 16, 16, 16, 24, 16, 32, 24, 32, 32, 40, 32, 32, 32 },
                    { 56,  48, 16, 10, 16, 24, 16, 24, 24, 24, 24, 24, 24, 24, 24 },
                },
            },
            { // f3x3, accuracy: 88.1%
                {
                    { 32, 48, 16, 16, 16, 16, 16, 24, 24, 32, 32, 24, 32, 32, 24 },
                    { 80, 24, 24, 16, 16, 16, 16, 16, 16, 24, 16, 16, 16, 24, 16 },
                    { 24, 16, 16, 16, 16, 16, 24, 16, 16, 24, 16, 24, 24, 24, 16 },
                    { 24, 16, 16, 16, 15, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16 },
                },
            },
        };

        constexpr DLimitsSet DLimitsSetNavi23 =
        {
            { // f1x1, accuracy: 94.0%
                {
                    { 208, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off },
                    { 112, 144, 384, 504, 472, 424, 392, 376, 344, 328, 312, 296, 296, 280, 272 },
                    { 104, 152, 128, Off, 304, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off },
                    { 88,  104, 72,  96,  64,  96,  64,  96,  64,  96,  64,  72,  64,  64,  64  },
                },
            },
            { // f2x2, accuracy: 93.1%
                {
                    { 56, 0,  16, 24, 24, 24, 24, 24, 24, 24, 24, 24, 16, 24, 24 },
                    { 40, 0,  10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 },
                    { 24, 0,  10, 14, 12, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16 },
                    { 24, 14, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14 },
                },
            },
            { // f3x3, accuracy: 90.2%
                {
                    { 0, 0, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 },
                    { 0, 0, 6,  13, 12, 12, 13, 11, 13, 12, 13, 11, 13, 13, 12 },
                    { 3, 0, 8,  10, 10, 9,  14, 10, 14, 14, 12, 12, 14, 14, 12 },
                    { 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 },
                },
            },
        };

        constexpr DLimitsSet DLimitsSetRmb =
        {
            { // f1x1, accuracy: 90.2%
                {
                    { 136, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off },
                    { 64,  120, 104, 96,  88,  88,  88,  80,  88,  80,  80,  64,  72,  72,  64  },
                    { 56,  112, 88,  Off, 264, Off, Off, Off, Off, Off, Off, Off, Off, Off, Off },
                    { 48,  80,  56,  80,  56,  64,  56,  56,  40,  56,  56,  56,  40,  80,  40  },
                },
            },
            { // f2x2, accuracy: 92.9%
                {
                    { 32, 24, 16, 16, 16, 16, 16, 24, 16, 16, 16, 16, 16, 16, 16 },
                    { 24, 15, 11, 13, 12, 15, 12, 16, 14, 13, 12, 13, 11, 13, 12 },
                    { 16, 16, 9,  12, 9,  13, 13, 14, 9,  13, 13, 14, 13, 13, 13 },
                    { 16, 16, 9,  9,  9,  10, 9,  13, 9,  9,  9,  9,  9,  9,  9  },
                },
            },
            { // f3x3, accuracy: 89.0%
                {
                    { 24, 16, 16, 32, 16, 16, 16, 16, 16, 13, 14, 15, 16, 12, 11 },
                    { 16, 11, 24, 16, 12, 11, 14, 7,  8,  7,  8,  8,  8,  8,  8  },
                    { 14, 9,  8,  24, 8,  9,  9,  6,  6,  8,  8,  7,  8,  9,  8  },
                    { 16, 9,  9,  14, 9,  8,  8,  9,  8,  6,  6,  7,  6,  6,  6  },
                },
            },
        };

        bool EvalDLimitsSet(
            const DLimitsSet&       limitSet,
            const GenericConvParams& params)
        {
            const uint32 rs = Min(params.r, params.s);

            const DLimits& limits = ((rs <= 1) ? limitSet.f1x1 :
                                     (rs <= 2) ? limitSet.f2x2 :
                                                 limitSet.f3x3);

            const uint32 ci = RoundUpQuotient(params.c, BinGranC) - 1;
            const uint32 ki = RoundUpQuotient(params.k, BinGranK) - 1;

            MLSS_ASSERT((ci < BinCountC) && (ki < BinCountK));

            const uint32 limitSqrt = limits.roots[ki][ci];
            const uint32 limit     = limitSqrt * limitSqrt;

            const float d = static_cast<float>(params.n) * params.outH * params.outW;

            return (limitSqrt != Off) && (d > limit);
        }

    } // namespace

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const GenericConvParams& params)
    {
        using mlss::op::utils::MetaCmdCaps;

        bool isArchSupported = gfxip == IP_GFX1030;
        bool isFp16 = params.dataType == DataTypeFlags::FLOAT16;
        bool isPrecisionFp16 = params.precision == PrecisionFlags::FLOAT16;
        bool isActivationSupported = (params.activation == ActivationFunctionFlags::RELU)
                                  || (params.activation == ActivationFunctionFlags::LEAKY_RELU)
                                  || (params.activation == ActivationFunctionFlags::IDENTITY)
                                  || (params.activation == ActivationFunctionFlags::SIGMOID);

        bool isSupported = isArchSupported && isFp16 && isPrecisionFp16 && isActivationSupported;
        bool isFullySupported = false;

        if ((params.c    > MaxC)                                  ||
            (params.k    > MaxK)                                  ||
            (params.r    > 0x03u)                                 ||
            (params.s    > 0x03u)                                 ||
            (params.n    > std::numeric_limits<std::uint16_t>::max()) ||
            (params.h    > std::numeric_limits<std::uint16_t>::max()) ||
            (params.w    > std::numeric_limits<std::uint16_t>::max()) ||
            (params.outW > std::numeric_limits<std::uint16_t>::max()) ||
            (params.outH > std::numeric_limits<std::uint16_t>::max()))
        {
            isSupported = false;
        }
        else if ((params.convStrideX   != 0x01u) ||
                 (params.convStrideY   != 0x01u) ||
                 (params.inputStrideX  != 0x01u) ||
                 (params.inputStrideY  != 0x01u) ||
                 (params.filterStrideX != 0x01u) ||
                 (params.filterStrideY != 0x01u) ||
                 (params.groups        != 0x01u))
        {
            isSupported = false;
        }
        else if ((params.outPadX > 0x00u) || (params.outPadY > 0x00u))
        {
            isSupported = false;
        }

        if (isSupported)
        {
            isSupported = ComputeCbStrides(params);
        }

        if (isSupported)
        {
            bool passRmb    = EvalDLimitsSet(DLimitsSetRmb, params);
            bool passNavi21 = EvalDLimitsSet(DLimitsSetNavi21, params);
            bool passNavi23 = EvalDLimitsSet(DLimitsSetNavi23, params);

            isFullySupported = isSupported && (passRmb || passNavi21 || passNavi23);
        }

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported;
        caps.fullSupport = isFullySupported;

        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const GenericConvParams& params)
    {
        auto capsResult = isShadersAvailable(gfxip, params);
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

} // namespace mlss::conv::mxn::winograd::ultra
