/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

namespace gfx1100 = mlss::mha::ck::wmma::fp16::gfx1100;
namespace gfx1150 = mlss::mha::ck::wmma::fp16::gfx1150;
namespace gfx1201 = mlss::mha::ck::wmma::fp16::gfx1201;
namespace fp16_constants = mlss::mha::ck::wmma::fp16;

namespace mlss::mha::ck::wmma
{
    using mlss::isGfx110x;
    using mlss::isGfx115x;
    using mlss::isGfx120x;

    namespace
    {

        enum class MHAAsmShaderWmma : std::uint32_t
        {
            unpacked_128_64x128x80_64x80x64,
            unpacked_128_64x192x48_64x48x64,
            unpacked_128_64x64x48_64x48x64,
            unpacked_fallback_64_32x64x48_32x48x64,

            packed_kv_128_64x128x80_64x80x64,
            packed_kv_128_64x192x48_64x48x64,
            packed_kv_128_64x64x48_64x48x64,
            packed_kv_fallback_64_32x64x48_32x48x64,

            packed_qk_128_64x128x80_64x80x64,
            packed_qk_128_64x192x48_64x48x64,
            packed_qk_128_64x64x48_64x48x64,
            packed_qk_fallback_64_32x64x48_32x48x64,

            packed_qkv_128_64x128x80_64x80x64,
            packed_qkv_128_64x192x48_64x48x64,
            packed_qkv_128_64x64x48_64x48x64,
            packed_qkv_fallback_64_32x64x48_32x48x64,

            count
        };

        template<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>
        std::pair<MLSSdim3, MLSSdim3> calcGridAndBlocks(
            const std::uint32_t& batchSize, const std::uint32_t& headCount,
            const std::uint32_t& kvSequenceLength, const std::uint32_t& qSequenceLength,
            const std::uint32_t& headDim)
        {
            std::ignore = kvSequenceLength;
            const auto M0 = integer_divide_ceil(qSequenceLength, MPerBlock);
            const auto N0 = integer_divide_ceil(headDim, NPerBlock);

            const auto grid_size = batchSize * headCount * M0 * N0;

            return std::make_pair(MLSSdim3(grid_size, 1, 1), MLSSdim3(BlockSize, 1, 1));
        }

        template<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>
        void populateBlobs(
            Binaries& binaries,
            const std::uint32_t& batchSize, const std::uint32_t& headCount,
            const std::uint32_t& kvSequenceLength, const std::uint32_t& qSequenceLength,
            const std::uint32_t& headDim,
            const auto& shader,
            const auto& constants, const auto& argConstants, const auto& argConstantsWithStrides)
        {
            auto [grid, blocks] = calcGridAndBlocks<MPerBlock, NPerBlock, BlockSize>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);

            Blob blob = std::move(*make_binary_blob(shader));
            blob = constants;
            blob = argConstants;
            blob.setGridBlocks(grid, blocks);

            Blob blobWithStrides = std::move(*make_binary_blob(shader));
            blobWithStrides = constants;
            blobWithStrides = argConstantsWithStrides;
            blobWithStrides.setGridBlocks(grid, blocks);

            binaries.addBlob(std::move(blob));
            binaries.addBlob(std::move(blobWithStrides));
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
        std::ignore = packing;

        if (!isGfx110x(gfxArch) && !isGfx115x(gfxArch) && !isGfx120x(gfxArch))
        {
            return false;
        }

        if (dataType != MLSS_FLOAT16)
        {
            return false;
        }

        return (sizeHeads <= 48) || ((sizeHeads % 2) == 0) || ((qSequenceLength == kvSequenceLength) && (sizeHeads <= 80));
    }

    std::expected<Binaries, std::error_code> getShadersBlob(
        GfxIpTriple gfxArch,
        const std::uint32_t& batchSize,
        const std::uint32_t& headCount,
        const std::uint32_t& headDim,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        const std::uint32_t& dataType)
    {
        if (!isShadersAvailable(gfxArch, headDim, kvSequenceLength, qSequenceLength, packing, dataType))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

        bool isSelfAttention = qSequenceLength == kvSequenceLength;
        bool isUnpacked = packing == MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
        bool isPackedKV = packing == MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_KV;
        bool isPackedQK = packing == MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_QK;
        bool isPackedQKV = packing == MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_QKV;

        MHAAsmShaderWmma shader = MHAAsmShaderWmma::count;

        if (isSelfAttention)
        {
            if (headDim <= 48)
            {
                if (isUnpacked)
                    shader = MHAAsmShaderWmma::unpacked_128_64x192x48_64x48x64;
                else if (isPackedKV)
                    shader = MHAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64;
                else if (isPackedQK)
                    shader = MHAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64;
                else if (isPackedQKV)
                    shader = MHAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64;
            }
            else if (headDim <= 80)
            {
                if (isUnpacked)
                    shader = MHAAsmShaderWmma::unpacked_128_64x128x80_64x80x64;
                else if (isPackedKV)
                    shader = MHAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64;
                else if (isPackedQK)
                    shader = MHAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64;
                else if (isPackedQKV)
                    shader = MHAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64;
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnpacked)
                    shader = MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64;
                else if (isPackedKV)
                    shader = MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64;
                else if (isPackedQK)
                    shader = MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64;
                else if (isPackedQKV)
                    shader = MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64;
            }
        }
        else
        {
            if (headDim <= 48)
            {
                if (isUnpacked)
                    shader = MHAAsmShaderWmma::unpacked_128_64x64x48_64x48x64;
                else if (isPackedKV)
                    shader = MHAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64;
                else if (isPackedQK)
                    shader = MHAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64;
                else if (isPackedQKV)
                    shader = MHAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64;
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnpacked)
                    shader = MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64;
                else if (isPackedKV)
                    shader = MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64;
                else if (isPackedQK)
                    shader = MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64;
                else if (isPackedQKV)
                    shader = MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64;
            }
        }

        if (shader == MHAAsmShaderWmma::count)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

        Binaries binaries;

#define MHA_DISPATCH(MPerBlock, NPerBlock, BlockSize, shaderRef, dim_constants, arg_no_strides, arg_with_strides) \
    populateBlobs<MPerBlock, NPerBlock, BlockSize>(binaries, batchSize, headCount, kvSequenceLength, qSequenceLength, headDim, \
        shaderRef, \
        fp16_constants::dim_constants, fp16_constants::arg_no_strides, fp16_constants::arg_with_strides)

#define MHA_ARCH_SWITCH(arch)                                                                                         \
    switch (shader)                                                                                                    \
    {                                                                                                                  \
        case MHAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:                                                      \
            MHA_DISPATCH(64,80,128, arch::unpacked_self_attention_128_64x128x80_64x80x64_forward_##arch,              \
                self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,                                              \
                unpacked_q_k_v_self_attention_ARGS_CONSTANT,                                                          \
                unpacked_q_k_v_with_strides_ARGS_CONSTANTS); break;                                                   \
        case MHAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:                                                      \
            MHA_DISPATCH(64,48,128, arch::unpacked_self_attention_128_64x192x48_64x48x64_forward_##arch,              \
                self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,                                              \
                unpacked_q_k_v_self_attention_ARGS_CONSTANT,                                                          \
                unpacked_q_k_v_with_strides_ARGS_CONSTANTS); break;                                                   \
        case MHAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:                                                       \
            MHA_DISPATCH(64,48,128, arch::unpacked_cross_attention_128_64x64x48_64x48x64_forward_##arch,              \
                cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,                                              \
                unpacked_q_k_v_cross_attention_ARGS_CONSTANT,                                                         \
                unpacked_q_k_v_with_strides_ARGS_CONSTANTS); break;                                                   \
        case MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64:                                               \
            MHA_DISPATCH(32,48,64, arch::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_##arch,       \
                fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,                                      \
                unpacked_q_k_v_cross_attention_ARGS_CONSTANT,                                                         \
                unpacked_q_k_v_with_strides_ARGS_CONSTANTS); break;                                                   \
        case MHAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:                                                     \
            MHA_DISPATCH(64,80,128, arch::self_attention_128_64x128x80_64x80x64_forward_##arch,                       \
                self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,                                              \
                packed_q_kv_cross_attention_ARGS_CONSTANTS,                                                            \
                packed_q_kv_with_strides_ARGS_CONSTANTS); break;                                                      \
        case MHAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:                                                     \
            MHA_DISPATCH(64,48,128, arch::self_attention_128_64x192x48_64x48x64_forward_##arch,                       \
                self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,                                              \
                packed_q_kv_cross_attention_ARGS_CONSTANTS,                                                            \
                packed_q_kv_with_strides_ARGS_CONSTANTS); break;                                                      \
        case MHAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:                                                      \
            MHA_DISPATCH(64,48,128, arch::cross_attention_128_64x64x48_64x48x64_forward_##arch,                       \
                cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,                                              \
                packed_q_kv_cross_attention_ARGS_CONSTANTS,                                                            \
                packed_q_kv_with_strides_ARGS_CONSTANTS); break;                                                      \
        case MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64:                                              \
            MHA_DISPATCH(32,48,64, arch::fallback_cross_attention_64_32x64x48_32x48x64_forward_##arch,                \
                fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,                                      \
                packed_q_kv_cross_attention_ARGS_CONSTANTS,                                                            \
                packed_q_kv_with_strides_ARGS_CONSTANTS); break;                                                      \
        case MHAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:                                                     \
            MHA_DISPATCH(64,80,128, arch::self_attention_128_64x128x80_64x80x64_forward_##arch,                       \
                self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,                                              \
                packed_qk_ARGS_CONSTANTS,                                                                              \
                packed_qk_with_strides_ARGS_CONSTANTS); break;                                                        \
        case MHAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:                                                     \
            MHA_DISPATCH(64,48,128, arch::self_attention_128_64x192x48_64x48x64_forward_##arch,                       \
                self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,                                              \
                packed_qk_ARGS_CONSTANTS,                                                                              \
                packed_qk_with_strides_ARGS_CONSTANTS); break;                                                        \
        case MHAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:                                                      \
            MHA_DISPATCH(64,48,128, arch::cross_attention_128_64x64x48_64x48x64_forward_##arch,                       \
                cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,                                              \
                packed_qk_ARGS_CONSTANTS,                                                                              \
                packed_qk_with_strides_ARGS_CONSTANTS); break;                                                        \
        case MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64:                                              \
            MHA_DISPATCH(32,48,64, arch::fallback_cross_attention_64_32x64x48_32x48x64_forward_##arch,                \
                fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,                                      \
                packed_qk_ARGS_CONSTANTS,                                                                              \
                packed_qk_with_strides_ARGS_CONSTANTS); break;                                                        \
        case MHAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:                                                    \
            MHA_DISPATCH(64,80,128, arch::self_attention_128_64x128x80_64x80x64_forward_##arch,                       \
                self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,                                              \
                packed_qkv_self_attention_ARGS_CONSTANTS,                                                               \
                packed_qkv_with_strides_ARGS_CONSTANTS); break;                                                       \
        case MHAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:                                                    \
            MHA_DISPATCH(64,48,128, arch::self_attention_128_64x192x48_64x48x64_forward_##arch,                       \
                self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,                                              \
                packed_qkv_self_attention_ARGS_CONSTANTS,                                                               \
                packed_qkv_with_strides_ARGS_CONSTANTS); break;                                                       \
        case MHAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:                                                     \
            MHA_DISPATCH(64,48,128, arch::cross_attention_128_64x64x48_64x48x64_forward_##arch,                       \
                cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,                                              \
                packed_qkv_self_attention_ARGS_CONSTANTS,                                                               \
                packed_qkv_with_strides_ARGS_CONSTANTS); break;                                                       \
        case MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64:                                             \
            MHA_DISPATCH(32,48,64, arch::fallback_cross_attention_64_32x64x48_32x48x64_forward_##arch,                \
                fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,                                      \
                packed_qkv_self_attention_ARGS_CONSTANTS,                                                               \
                packed_qkv_with_strides_ARGS_CONSTANTS); break;                                                       \
        default:                                                                                                       \
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));                         \
    }

        if (isGfx110x(gfxArch))
        {
            MHA_ARCH_SWITCH(gfx1100)
        }
        else if (isGfx115x(gfxArch))
        {
            MHA_ARCH_SWITCH(gfx1150)
        }
        else if (isGfx120x(gfxArch))
        {
            MHA_ARCH_SWITCH(gfx1201)
        }
        else
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

#undef MHA_ARCH_SWITCH
#undef MHA_DISPATCH

        return binaries;
    }

} // namespace mlss::mha::ck::wmma
