/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

#include <mutex>
#include <unordered_map>

namespace gfx1100 = mlss::gqa::ck::wmma::fp16::gfx1100;
namespace gfx1150 = mlss::gqa::ck::wmma::fp16::gfx1150;
namespace gfx1201 = mlss::gqa::ck::wmma::fp16::gfx1201;
namespace fp16_constants = mlss::gqa::ck::wmma::fp16;

namespace mlss::gqa::ck::wmma
{
    using mlss::isGfx110x;
    using mlss::isGfx115x;
    using mlss::isGfx120x;

    namespace
    {
        enum class GQAAsmShaderWmma : std::uint32_t
        {
            packed_qk_128_64x128x80_64x80x64,
            packed_qk_128_64x192x48_64x48x64,
            packed_qk_128_64x64x48_64x48x64,
            packed_qk_64_32x64x48_32x48x64,

            packed_kv_128_64x128x80_64x80x64,
            packed_kv_128_64x192x48_64x48x64,
            packed_kv_128_64x64x48_64x48x64,
            packed_kv_64_32x64x48_32x48x64,

            packed_qkv_128_64x128x80_64x80x64,
            packed_qkv_128_64x192x48_64x48x64,
            packed_qkv_128_64x64x48_64x48x64,
            packed_qkv_64_32x64x48_32x48x64,

            unpacked_128_64x128x80_64x80x64,
            unpacked_128_64x192x48_64x48x64,
            unpacked_128_64x64x48_64x48x64,
            unpacked_64_32x64x48_32x48x64,

            count
        };

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x00u) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x05u) return {0x0Bu, 0x05u, 0x00u};
            if (gfxip.major == 0x0Cu)                         return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        std::uint64_t makeCacheKey(const GfxIpTriple& gfxip, GQAAsmShaderWmma shaderEnum)
        {
            return (static_cast<std::uint64_t>(gfxIpPacked(gfxip)) << 0x20u)
                 | static_cast<std::uint64_t>(shaderEnum);
        }

        std::mutex s_cacheMutex;
        std::unordered_map<std::uint64_t, DynamicShaderType> s_shaderCache;

        template<typename ShaderT>
        const DynamicShaderType* getOrComputeCached(
            const GfxIpTriple& gfxip,
            GQAAsmShaderWmma shaderEnum,
            const ShaderT& shader)
        {
            auto key = makeCacheKey(gfxip, shaderEnum);

            {
                std::lock_guard lock(s_cacheMutex);
                auto it = s_shaderCache.find(key);
                if (it != s_shaderCache.end())
                {
                    return &it->second;
                }
            }

            auto relocDescriptor = make_shader_descriptor(shader);
            if (relocDescriptor.m_binary.empty())
            {
                return nullptr;
            }

            auto sourceArch = sourceArchForTarget(gfxip);
            auto nonRelocResult = getNonRelocatable(relocDescriptor.m_binary, sourceArch, gfxip);
            if (!nonRelocResult.has_value())
            {
                return nullptr;
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

        template<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>
        std::pair<MLSSdim3, MLSSdim3> calcGridAndBlocks(
            const std::uint32_t& batchSize, const std::uint32_t& headCount,
            const std::uint32_t& kvSequenceLength, const std::uint32_t& qSequenceLength,
            const std::uint32_t& headDim)
        {
            const auto M0 = integer_divide_ceil(qSequenceLength, MPerBlock);
            const auto N0 = integer_divide_ceil(headDim, NPerBlock);

            const auto grid_size = batchSize * headCount * M0 * N0;

            return std::make_pair(MLSSdim3(grid_size, 1, 1), MLSSdim3(BlockSize, 1, 1));
        }

        template<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>
        void populateBlobs(
            Binaries& binaries,
            const GfxIpTriple& gfxArch,
            GQAAsmShaderWmma shaderEnum,
            const std::uint32_t& batchSize, const std::uint32_t& qHeadCount,
            const std::uint32_t& kvSequenceLength, const std::uint32_t& qSequenceLength,
            const std::uint32_t& headDim,
            const auto& shader,
            const auto& constants, const auto& argConstantsNoStrides, const auto& argConstantsWithStrides)
        {
            auto [grid, blocks] = calcGridAndBlocks<MPerBlock, NPerBlock, BlockSize>(batchSize, qHeadCount, kvSequenceLength, qSequenceLength, headDim);

            Blob blob = std::move(*make_binary_blob(shader));
            blob = constants;
            blob = argConstantsNoStrides;
            blob.setGridBlocks(grid, blocks);

            Blob blobWithStrides = std::move(*make_binary_blob(shader));
            blobWithStrides = constants;
            blobWithStrides = argConstantsWithStrides;
            blobWithStrides.setGridBlocks(grid, blocks);

            binaries.addBlob(std::move(blob));
            binaries.addBlob(std::move(blobWithStrides));

            const auto* cachedShader = getOrComputeCached(gfxArch, shaderEnum, shader);
            if (cachedShader)
            {
                auto nonRelocDescriptor = make_shader_descriptor(*cachedShader);

                Blob nonRelocBlob = std::move(*make_binary_blob(nonRelocDescriptor));
                nonRelocBlob = constants;
                nonRelocBlob = argConstantsNoStrides;
                nonRelocBlob.setGridBlocks(grid, blocks);

                Blob nonRelocBlobStrides = std::move(*make_binary_blob(nonRelocDescriptor));
                nonRelocBlobStrides = constants;
                nonRelocBlobStrides = argConstantsWithStrides;
                nonRelocBlobStrides.setGridBlocks(grid, blocks);

                binaries.addBlob(std::move(nonRelocBlob));
                binaries.addBlob(std::move(nonRelocBlobStrides));
            }
        }

    } // anonymous namespace

    bool isShadersAvailable(
        GfxIpTriple gfxArch,
        const std::uint32_t& sizeHeads,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        const std::uint32_t& dataType)
    {
        if (!isGfx110x(gfxArch) && !isGfx115x(gfxArch) && !isGfx120x(gfxArch))
        {
            return false;
        }

        if (dataType != MLSS_FLOAT16)
        {
            return false;
        }

        const bool isSelfAttention = qSequenceLength == kvSequenceLength;

        // Reject packings that have no compiled CK kernel matching the advertised args layout:
        //   - PACKED_QK: no `packed_qk_gqa_*` kernel exists in any per-arch shadersBin.hpp; the
        //     `packed_kv_gqa_*` kernels we previously fell back to expose (Q, KV, ...) buffers,
        //     not the (QK, V, ...) layout advertised by the args metadata.
        //   - PACKED_QKV cross-attention: no `packed_qkv_gqa_cross_attn_*` /
        //     `packed_qkv_gqa_fallback_cross_attn_*` kernel exists, and falling back to a
        //     `packed_kv_gqa_cross_attn_*` kernel introduces a hard arg-count mismatch
        //     (10 vs 8, 22 vs 16).
        if (packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QK)
        {
            return false;
        }
        if ((packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QKV) && !isSelfAttention)
        {
            return false;
        }

        return (sizeHeads <= 48) || ((sizeHeads % 2) == 0) || (isSelfAttention && (sizeHeads <= 80));
    }

    std::expected<Binaries, std::error_code> getShadersBlob(
        GfxIpTriple gfxArch,
        const std::uint32_t& batchSize,
        const std::uint32_t& qHeadCount,
        const std::uint32_t& kvHeadCount,
        const std::uint32_t& headDim,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        bool useStrides,
        const std::uint32_t& dataType)
    {
        if (!isShadersAvailable(gfxArch, headDim, kvSequenceLength, qSequenceLength, packing, dataType))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

        bool isSelfAttention = qSequenceLength == kvSequenceLength;
        bool isUnPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_UNPACKED;
        bool isQKPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QK;
        bool isQVPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_KV;
        bool isQKVPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QKV;

        GQAAsmShaderWmma shader = GQAAsmShaderWmma::count;

        if (isSelfAttention)
        {
            if (headDim <= 48)
            {
                if (isUnPacked)
                {
                    shader = GQAAsmShaderWmma::unpacked_128_64x192x48_64x48x64;
                }
                else if (isQKPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64;
                }
                else if (isQVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64;
                }
                else if (isQKVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64;
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }
            }
            else if (headDim <= 80)
            {
                if (isUnPacked)
                {
                    shader = GQAAsmShaderWmma::unpacked_128_64x128x80_64x80x64;
                }
                else if (isQKPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64;
                }
                else if (isQVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64;
                }
                else if (isQKVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64;
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnPacked)
                {
                    shader = GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64;
                }
                else if (isQKPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qk_64_32x64x48_32x48x64;
                }
                else if (isQVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64;
                }
                else if (isQKVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64;
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }
            }
        }
        else
        {
            if (headDim <= 48)
            {
                if (isUnPacked)
                {
                    shader = GQAAsmShaderWmma::unpacked_128_64x64x48_64x48x64;
                }
                else if (isQKPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64;
                }
                else if (isQVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64;
                }
                else if (isQKVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64;
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnPacked)
                {
                    shader = GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64;
                }
                else if (isQKPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qk_64_32x64x48_32x48x64;
                }
                else if (isQVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64;
                }
                else if (isQKVPacked)
                {
                    shader = GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64;
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }
            }
            else
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
            }
        }

        Binaries binaries;

#define GQA_DISPATCH(MPerBlock, NPerBlock, BlockSize, shaderRef, dim_constants, arg_no_strides, arg_with_strides) \
    populateBlobs<MPerBlock, NPerBlock, BlockSize>(binaries, gfxArch, shader, batchSize, qHeadCount, kvSequenceLength, qSequenceLength, headDim, \
        shaderRef, \
        fp16_constants::dim_constants, fp16_constants::arg_no_strides, fp16_constants::arg_with_strides)

#define GQA_ARCH_SWITCH(arch)                                                                                         \
    switch (shader)                                                                                                    \
    {                                                                                                                  \
        case GQAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:                                                     \
            GQA_DISPATCH(64,80,128, arch::packed_kv_gqa_self_attn_128_64x128x80_64x80x64_forward_wmma_fp16_##arch,   \
                gqa_128_64x128x80_64x80x64_CONSTANTS,                                                                \
                packed_kv_double_pointer_ARGS_CONSTANTS,                                                              \
                packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                         \
        case GQAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:                                                     \
            GQA_DISPATCH(64,48,128, arch::packed_kv_gqa_self_attn_128_64x192x48_64x48x64_forward_wmma_fp16_##arch,   \
                gqa_128_64x192x48_64x48x64_CONSTANTS,                                                                \
                packed_kv_double_pointer_ARGS_CONSTANTS,                                                              \
                packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                         \
        case GQAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:                                                      \
            GQA_DISPATCH(64,48,128, arch::packed_kv_gqa_cross_attn_128_64x64x48_64x48x64_forward_wmma_fp16_##arch,   \
                gqa_128_64x64x48_64x48x64_CONSTANTS,                                                                 \
                packed_kv_double_pointer_ARGS_CONSTANTS,                                                              \
                packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                         \
        case GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64:                                                       \
            GQA_DISPATCH(32,48,64, arch::packed_kv_gqa_fallback_cross_attn_64_32x64x48_32x48x64_forward_wmma_fp16_##arch, \
                gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,                                                         \
                packed_kv_double_pointer_ARGS_CONSTANTS,                                                              \
                packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                         \
        case GQAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:                                                    \
            GQA_DISPATCH(64,80,128, arch::packed_qkv_gqa_self_attn_128_64x128x80_64x80x64_forward_wmma_fp16_##arch,  \
                gqa_128_64x128x80_64x80x64_CONSTANTS,                                                                \
                packed_qkv_double_pointer_ARGS_CONSTANTS,                                                             \
                packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                        \
        case GQAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:                                                    \
            GQA_DISPATCH(64,48,128, arch::packed_qkv_gqa_self_attn_128_64x192x48_64x48x64_forward_wmma_fp16_##arch,  \
                gqa_128_64x192x48_64x48x64_CONSTANTS,                                                                \
                packed_qkv_double_pointer_ARGS_CONSTANTS,                                                             \
                packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                        \
        case GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64:                                                      \
            GQA_DISPATCH(32,48,64, arch::packed_qkv_gqa_fallback_self_attn_64_32x64x48_32x48x64_forward_wmma_fp16_##arch, \
                gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,                                                         \
                packed_qkv_double_pointer_ARGS_CONSTANTS,                                                             \
                packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break;                                        \
        case GQAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:                                                      \
            GQA_DISPATCH(64,80,128, arch::unpacked_gqa_self_attn_128_64x128x80_64x80x64_forward_wmma_fp16_##arch,    \
                gqa_128_64x128x80_64x80x64_CONSTANTS,                                                                \
                unpacked_double_pointer_ARGS_CONSTANTS,                                                               \
                unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break;                                          \
        case GQAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:                                                      \
            GQA_DISPATCH(64,48,128, arch::unpacked_gqa_self_attn_128_64x192x48_64x48x64_forward_wmma_fp16_##arch,    \
                gqa_128_64x192x48_64x48x64_CONSTANTS,                                                                \
                unpacked_double_pointer_ARGS_CONSTANTS,                                                               \
                unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break;                                          \
        case GQAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:                                                       \
            GQA_DISPATCH(64,48,128, arch::unpacked_gqa_cross_attn_128_64x64x48_64x48x64_forward_wmma_fp16_##arch,    \
                gqa_128_64x64x48_64x48x64_CONSTANTS,                                                                 \
                unpacked_double_pointer_ARGS_CONSTANTS,                                                               \
                unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break;                                          \
        case GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64:                                                        \
            GQA_DISPATCH(32,48,64, arch::unpacked_gqa_fallback_cross_attn_64_32x64x48_32x48x64_forward_wmma_fp16_##arch, \
                gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,                                                         \
                unpacked_double_pointer_ARGS_CONSTANTS,                                                               \
                unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break;                                          \
        default:                                                                                                       \
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));                         \
    }

        if (isGfx110x(gfxArch))
        {
            GQA_ARCH_SWITCH(gfx1100)
        }
        else if (isGfx115x(gfxArch))
        {
            GQA_ARCH_SWITCH(gfx1150)
        }
        else if (isGfx120x(gfxArch))
        {
            GQA_ARCH_SWITCH(gfx1201)
        }
        else
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

#undef GQA_ARCH_SWITCH
#undef GQA_DISPATCH

        return binaries;
    }

} // namespace mlss::gqa::ck::wmma
