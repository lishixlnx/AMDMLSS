/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

import grouped_query_attention_shaders_gfx1100_reloc;
import grouped_query_attention_shaders_gfx1100_non_reloc;
import grouped_query_attention_shaders_gfx1150_reloc;
import grouped_query_attention_shaders_gfx1150_non_reloc;
import grouped_query_attention_shaders_gfx1201_reloc;
import grouped_query_attention_shaders_gfx1201_non_reloc;

namespace mlss::gqa::ck::wmma
{
    namespace fp16
    {
        namespace gfx1100
        {
            namespace rel = ::gqa::wmma::gfx1100::rel;
            namespace nonrel = ::gqa::wmma::gfx1100::nonrel;
        } // namespace gfx1100
        namespace gfx1150
        {
            namespace rel = ::gqa::wmma::gfx1150::rel;
            namespace nonrel = ::gqa::wmma::gfx1150::nonrel;
        } // namespace gfx1150
        namespace gfx1201
        {
            namespace rel = ::gqa::wmma::gfx1201::rel;
            namespace nonrel = ::gqa::wmma::gfx1201::nonrel;
        } // namespace gfx1201
    } // namespace fp16
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
            const std::uint32_t& batchSize, const std::uint32_t& qHeadCount,
            const std::uint32_t& kvSequenceLength, const std::uint32_t& qSequenceLength,
            const std::uint32_t& headDim,
            const auto& shaderRel, const auto& shaderNonRel,
            const auto& shaderRelWithStrides, const auto& shaderNonRelWithStrides,
            const auto& constants, const auto& argConstantsNoStrides, const auto& argConstantsWithStrides)
        {
            auto [grid, blocks] = calcGridAndBlocks<MPerBlock, NPerBlock, BlockSize>(batchSize, qHeadCount, kvSequenceLength, qSequenceLength, headDim);

            Blob relBlob = std::move(*make_binary_blob(shaderRel));
            relBlob = constants;
            relBlob = argConstantsNoStrides;
            relBlob.setGridBlocks(grid, blocks);

            Blob nonRelBlob = std::move(*make_binary_blob(shaderNonRel));
            nonRelBlob = constants;
            nonRelBlob = argConstantsNoStrides;
            nonRelBlob.setGridBlocks(grid, blocks);

            Blob relBlobWithStrides = std::move(*make_binary_blob(shaderRelWithStrides));
            relBlobWithStrides = constants;
            relBlobWithStrides = argConstantsWithStrides;
            relBlobWithStrides.setGridBlocks(grid, blocks);

            Blob nonRelBlobWithStrides = std::move(*make_binary_blob(shaderNonRelWithStrides));
            nonRelBlobWithStrides = constants;
            nonRelBlobWithStrides = argConstantsWithStrides;
            nonRelBlobWithStrides.setGridBlocks(grid, blocks);

            binaries.addBlob(std::move(relBlob));
            binaries.addBlob(std::move(nonRelBlob));
            binaries.addBlob(std::move(relBlobWithStrides));
            binaries.addBlob(std::move(nonRelBlobWithStrides));
        }

    } // anonymous namespace

    bool isShadersAvailable(
        GfxIpTriple gfxArch,
        const std::uint32_t& sizeHeads,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
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

        return (sizeHeads <= 48) || ((sizeHeads % 2) == 0) || ((qSequenceLength == kvSequenceLength) && (sizeHeads <= 80));
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
        if (!isShadersAvailable(gfxArch, headDim, kvSequenceLength, qSequenceLength, dataType))
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

#define GQA_DISPATCH(MPerBlock, NPerBlock, BlockSize, ns, variant, dim_constants, arg_no_strides, arg_with_strides) \
    populateBlobs<MPerBlock, NPerBlock, BlockSize>(binaries, batchSize, qHeadCount, kvSequenceLength, qSequenceLength, headDim, \
        fp16::ns::rel::grouped_query_attention_typed_double_pointer_##variant##_forward_wmma_no_strides_fp16_##ns, \
        fp16::ns::nonrel::grouped_query_attention_typed_double_pointer_##variant##_forward_wmma_no_strides_fp16_##ns, \
        fp16::ns::rel::grouped_query_attention_typed_double_pointer_##variant##_forward_wmma_with_strides_fp16_##ns, \
        fp16::ns::nonrel::grouped_query_attention_typed_double_pointer_##variant##_forward_wmma_with_strides_fp16_##ns, \
        fp16::dim_constants, fp16::arg_no_strides, fp16::arg_with_strides)

#define GQA_ARCH_SWITCH(ns)                                                                                           \
    switch (shader)                                                                                                    \
    {                                                                                                                  \
        case GQAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:                                                     \
            GQA_DISPATCH(64,80,128, ns, packed_qk_128_64x128x80_64x80x64, gqa_128_64x128x80_64x80x64_CONSTANTS, packed_qk_double_pointer_ARGS_CONSTANTS, packed_qk_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:                                                     \
            GQA_DISPATCH(64,48,128, ns, packed_qk_128_64x192x48_64x48x64, gqa_128_64x192x48_64x48x64_CONSTANTS, packed_qk_double_pointer_ARGS_CONSTANTS, packed_qk_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:                                                      \
            GQA_DISPATCH(64,48,128, ns, packed_qk_128_64x64x48_64x48x64, gqa_128_64x64x48_64x48x64_CONSTANTS, packed_qk_double_pointer_ARGS_CONSTANTS, packed_qk_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qk_64_32x64x48_32x48x64:                                                       \
            GQA_DISPATCH(32,48,64, ns, packed_qk_fallback_64_32x64x48_32x48x64, gqa_fallback_64_32x64x48_32x48x64_CONSTANTS, packed_qk_double_pointer_ARGS_CONSTANTS, packed_qk_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:                                                     \
            GQA_DISPATCH(64,80,128, ns, packed_kv_128_64x128x80_64x80x64, gqa_128_64x128x80_64x80x64_CONSTANTS, packed_kv_double_pointer_ARGS_CONSTANTS, packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:                                                     \
            GQA_DISPATCH(64,48,128, ns, packed_kv_128_64x192x48_64x48x64, gqa_128_64x192x48_64x48x64_CONSTANTS, packed_kv_double_pointer_ARGS_CONSTANTS, packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:                                                      \
            GQA_DISPATCH(64,48,128, ns, packed_kv_128_64x64x48_64x48x64, gqa_128_64x64x48_64x48x64_CONSTANTS, packed_kv_double_pointer_ARGS_CONSTANTS, packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64:                                                       \
            GQA_DISPATCH(32,48,64, ns, packed_kv_fallback_64_32x64x48_32x48x64, gqa_fallback_64_32x64x48_32x48x64_CONSTANTS, packed_kv_double_pointer_ARGS_CONSTANTS, packed_kv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:                                                    \
            GQA_DISPATCH(64,80,128, ns, packed_qkv_128_64x128x80_64x80x64, gqa_128_64x128x80_64x80x64_CONSTANTS, packed_qkv_double_pointer_ARGS_CONSTANTS, packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:                                                    \
            GQA_DISPATCH(64,48,128, ns, packed_qkv_128_64x192x48_64x48x64, gqa_128_64x192x48_64x48x64_CONSTANTS, packed_qkv_double_pointer_ARGS_CONSTANTS, packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:                                                     \
            GQA_DISPATCH(64,48,128, ns, packed_qkv_128_64x64x48_64x48x64, gqa_128_64x64x48_64x48x64_CONSTANTS, packed_qkv_double_pointer_ARGS_CONSTANTS, packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64:                                                      \
            GQA_DISPATCH(32,48,64, ns, packed_qkv_fallback_64_32x64x48_32x48x64, gqa_fallback_64_32x64x48_32x48x64_CONSTANTS, packed_qkv_double_pointer_ARGS_CONSTANTS, packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:                                                      \
            GQA_DISPATCH(64,80,128, ns, unpacked_128_64x128x80_64x80x64, gqa_128_64x128x80_64x80x64_CONSTANTS, unpacked_double_pointer_ARGS_CONSTANTS, unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:                                                      \
            GQA_DISPATCH(64,48,128, ns, unpacked_128_64x192x48_64x48x64, gqa_128_64x192x48_64x48x64_CONSTANTS, unpacked_double_pointer_ARGS_CONSTANTS, unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:                                                       \
            GQA_DISPATCH(64,48,128, ns, unpacked_128_64x64x48_64x48x64, gqa_128_64x64x48_64x48x64_CONSTANTS, unpacked_double_pointer_ARGS_CONSTANTS, unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break; \
        case GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64:                                                        \
            GQA_DISPATCH(32,48,64, ns, unpacked_fallback_64_32x64x48_32x48x64, gqa_fallback_64_32x64x48_32x48x64_CONSTANTS, unpacked_double_pointer_ARGS_CONSTANTS, unpacked_with_strides_double_pointer_ARGS_CONSTANTS); break; \
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
