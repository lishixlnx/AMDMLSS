/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

import grouped_query_attention_shaders_gfx1100_reloc;
import grouped_query_attention_shaders_gfx1100_non_reloc;
import grouped_query_attention_shaders_gfx1150_reloc;
import grouped_query_attention_shaders_gfx1150_non_reloc;
import grouped_query_attention_shaders_gfx1201_reloc;
import grouped_query_attention_shaders_gfx1201_non_reloc;

namespace mlss::shaders::gqa::ck::wmma
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
        // GQA shader variants for CK implementation
        enum class GQAAsmShaderWmma : std::uint32_t
        {
            // Packed_qk double pointer variants
            packed_qk_128_64x128x80_64x80x64,
            packed_qk_128_64x192x48_64x48x64,
            packed_qk_128_64x64x48_64x48x64,
            packed_qk_64_32x64x48_32x48x64,

            // Packed_kv double pointer variants
            packed_kv_128_64x128x80_64x80x64,
            packed_kv_128_64x192x48_64x48x64,
            packed_kv_128_64x64x48_64x48x64,
            packed_kv_64_32x64x48_32x48x64,

            // Packed_qkv double pointer variants
            packed_qkv_128_64x128x80_64x80x64,
            packed_qkv_128_64x192x48_64x48x64,
            packed_qkv_128_64x64x48_64x48x64,
            packed_qkv_64_32x64x48_32x48x64,

            // Unpacked double pointer variants
            unpacked_128_64x128x80_64x80x64,
            unpacked_128_64x192x48_64x48x64,
            unpacked_128_64x64x48_64x48x64,
            unpacked_64_32x64x48_32x48x64,

            count
        };
    } // anonymous namespace

    bool isWmmaShadersAvailable(
        GfxArchitectureFlags gfxArch,
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

        // Check head dimension constraints
        return (sizeHeads <= 48) || ((sizeHeads % 2) == 0) || ((qSequenceLength == kvSequenceLength) && (sizeHeads <= 80));
    }

    std::expected<Binaries, std::error_code> getWmmaShadersBlob(
        GfxArchitectureFlags gfxArch,
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
        constexpr auto calcGridAndBlocks = []<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>(
                                               const std::uint32_t& batchSize, const std::uint32_t& headCount, const std::uint32_t& kvSequenceLength, const std::uint32_t& qSequenceLength, const std::uint32_t& headDim) -> std::pair<MLSSdim3, MLSSdim3>
        {
            const auto M0 = integer_divide_ceil(qSequenceLength, MPerBlock);
            const auto N0 = integer_divide_ceil(headDim, NPerBlock);

            const auto grid_size = batchSize * headCount * M0 * N0;

            return std::make_pair(MLSSdim3(grid_size, 1, 1), MLSSdim3(BlockSize, 1, 1));
        };

        if (!isWmmaShadersAvailable(gfxArch, headDim, kvSequenceLength, qSequenceLength, dataType))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

        bool isSelfAttention = qSequenceLength == kvSequenceLength;
        bool isUnPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_UNPACKED;
        bool isQKPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QK;
        bool isQVPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_KV;
        bool isQKVPacked = packing == MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QKV;

        GQAAsmShaderWmma shader = GQAAsmShaderWmma::count;

        // Select shader based on attention type and head dimensions
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

        auto setBlobs = [batchSize, qHeadCount, kvSequenceLength, qSequenceLength, headDim, &binaries, &calcGridAndBlocks]<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>(const auto& shaderRel, const auto& shaderNonRel, const auto& shaderRelWithStrides, const auto& shaderNonRelWithStrides, const auto& constants, const auto& arg_constants_no_strides, const auto& arg_constants_with_strides)
            -> void
        {
            Blob relBlob;
            Blob nonRelBlob;
            Blob relBlobWithStrides;
            Blob nonRelBlobWithStrides;

            auto [grid, blocks] = calcGridAndBlocks.template operator()<MPerBlock, NPerBlock, BlockSize>(batchSize, qHeadCount, kvSequenceLength, qSequenceLength, headDim);
            relBlob = std::move(*make_binary_blob(shaderRel));
            relBlob = constants;
            relBlob = arg_constants_no_strides;
            relBlob.setGridBlocks(grid, blocks);
            nonRelBlob = std::move(*make_binary_blob(shaderNonRel));
            nonRelBlob = constants;
            nonRelBlob = arg_constants_no_strides;
            nonRelBlob.setGridBlocks(grid, blocks);
            relBlobWithStrides = std::move(*make_binary_blob(shaderRelWithStrides));
            relBlobWithStrides = constants;
            relBlobWithStrides = arg_constants_with_strides;
            relBlobWithStrides.setGridBlocks(grid, blocks);
            nonRelBlobWithStrides = std::move(*make_binary_blob(shaderNonRelWithStrides));
            nonRelBlobWithStrides = constants;
            nonRelBlobWithStrides = arg_constants_with_strides;
            nonRelBlobWithStrides.setGridBlocks(grid, blocks);

            binaries.addBlob(std::move(relBlob));
            binaries.addBlob(std::move(nonRelBlob));
            binaries.addBlob(std::move(relBlobWithStrides));
            binaries.addBlob(std::move(nonRelBlobWithStrides));
        };

        if (isGfx110x(gfxArch))
        {
            switch (shader)
            {
                case GQAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1100,
                        fp16::gfx1100::rel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gfx1100::nonrel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1100,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
            }
        }
        else if (isGfx115x(gfxArch))
        {
            switch (shader)
            {
                case GQAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1150,
                        fp16::gfx1150::rel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gfx1150::nonrel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1150,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
            }
        }
        else if (isGfx120x(gfxArch))
        {
            switch (shader)
            {
                case GQAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qk_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_qk_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qk_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_kv_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_kv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_kv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::packed_qkv_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::packed_qkv_double_pointer_ARGS_CONSTANTS,
                        fp16::packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:
                {
                    setBlobs.template operator()<64, 80, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x128x80_64x80x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x192x48_64x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:
                {
                    setBlobs.template operator()<64, 48, 128>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_128_64x64x48_64x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                case GQAAsmShaderWmma::unpacked_64_32x64x48_32x48x64:
                {
                    setBlobs.template operator()<32, 48, 64>(
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_no_strides_fp16_gfx1201,
                        fp16::gfx1201::rel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gfx1201::nonrel::grouped_query_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_wmma_with_strides_fp16_gfx1201,
                        fp16::gqa_fallback_64_32x64x48_32x48x64_CONSTANTS,
                        fp16::unpacked_double_pointer_ARGS_CONSTANTS,
                        fp16::unpacked_with_strides_double_pointer_ARGS_CONSTANTS);
                    break;
                }
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
            }
        }
        else
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
        }

        return binaries;
    }

} // namespace mlss::shaders::gqa::ck::wmma
