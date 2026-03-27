/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

import multi_head_attention_shaders_gfx1100_reloc;
import multi_head_attention_shaders_gfx1100_non_reloc;
import multi_head_attention_shaders_gfx1150_reloc;
import multi_head_attention_shaders_gfx1150_non_reloc;
import multi_head_attention_shaders_gfx1201_reloc;
import multi_head_attention_shaders_gfx1201_non_reloc;

namespace gfx1100_rel    = ::mha::wmma::gfx1100::rel;
namespace gfx1100_nonrel = ::mha::wmma::gfx1100::nonrel;
namespace gfx1150_rel    = ::mha::wmma::gfx1150::rel;
namespace gfx1150_nonrel = ::mha::wmma::gfx1150::nonrel;
namespace gfx1201_rel    = ::mha::wmma::gfx1201::rel;
namespace gfx1201_nonrel = ::mha::wmma::gfx1201::nonrel;
namespace fp16_constants = ::mlss::shaders::mha::ck::wmma::fp16;

namespace mlss::shaders::mha::ck::wmma
{
    using mlss::isGfx110x;
    using mlss::isGfx115x;
    using mlss::isGfx120x;

    namespace 
    {

        // MHA shader variants for CK implementation
        enum class MHAAsmShaderWmma : std::uint32_t
        {
            // Unpacked variants (no strides)
            unpacked_128_64x128x80_64x80x64,
            unpacked_128_64x192x48_64x48x64,
            unpacked_128_64x64x48_64x48x64,
            unpacked_fallback_64_32x64x48_32x48x64,

            // Packed KV variants (no strides)
            packed_kv_128_64x128x80_64x80x64,
            packed_kv_128_64x192x48_64x48x64,
            packed_kv_128_64x64x48_64x48x64,
            packed_kv_fallback_64_32x64x48_32x48x64,

            // Packed QK variants (no strides)
            packed_qk_128_64x128x80_64x80x64,
            packed_qk_128_64x192x48_64x48x64,
            packed_qk_128_64x64x48_64x48x64,
            packed_qk_fallback_64_32x64x48_32x48x64,

            // Packed QKV variants (no strides)
            packed_qkv_128_64x128x80_64x80x64,
            packed_qkv_128_64x192x48_64x48x64,
            packed_qkv_128_64x64x48_64x48x64,
            packed_qkv_fallback_64_32x64x48_32x48x64,

            count
        };

    } // anonymous namespace

    bool isWmmaShadersAvailable(
        GfxArchitectureFlags gfxArch,
        const std::uint32_t& sizeHeads,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        const std::uint32_t& dataType)
    {
        std::ignore = packing;
        
        if(!isGfx110x(gfxArch) && !isGfx115x(gfxArch) && !isGfx120x(gfxArch))
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
        const std::uint32_t& headCount,
        const std::uint32_t& headDim,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        const std::uint32_t& dataType)
    {
        constexpr auto calcGridAndBlocks = []<std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>(
            const std::uint32_t& batchSize, const std::uint32_t& headCount, const std::uint32_t& kvSequenceLength,
            const std::uint32_t& qSequenceLength, const std::uint32_t& headDim) -> std::pair<MLSSdim3, MLSSdim3>
        {
            std::ignore = kvSequenceLength;
            const auto M0 = integer_divide_ceil(qSequenceLength, MPerBlock);
            const auto N0 = integer_divide_ceil(headDim, NPerBlock);

            const auto grid_size = batchSize * headCount * M0 * N0;

            return std::make_pair(MLSSdim3(grid_size, 1, 1), MLSSdim3(BlockSize, 1, 1));
        };
        
        if(!isWmmaShadersAvailable(gfxArch, headDim, kvSequenceLength, qSequenceLength, packing, dataType))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

        bool isSelfAttention = qSequenceLength == kvSequenceLength;
        bool isUnpacked = packing == MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
        bool isPackedKV = packing == MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_KV;
        bool isPackedQK = packing == MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_QK;
        bool isPackedQKV = packing == MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_QKV;

        MHAAsmShaderWmma shader = MHAAsmShaderWmma::count;

        // Select shader based on attention type, packing, and head dimensions
        if (isSelfAttention)
        {
            if (headDim <= 48)
            {
                if (isUnpacked) shader = MHAAsmShaderWmma::unpacked_128_64x192x48_64x48x64;
                else if (isPackedKV) shader = MHAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64;
                else if (isPackedQK) shader = MHAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64;
                else if (isPackedQKV) shader = MHAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64;
            }
            else if (headDim <= 80)
            {
                if (isUnpacked) shader = MHAAsmShaderWmma::unpacked_128_64x128x80_64x80x64;
                else if (isPackedKV) shader = MHAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64;
                else if (isPackedQK) shader = MHAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64;
                else if (isPackedQKV) shader = MHAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64;
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnpacked) shader = MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64;
                else if (isPackedKV) shader = MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64;
                else if (isPackedQK) shader = MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64;
                else if (isPackedQKV) shader = MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64;
            }
        }
        else // Cross attention
        {
            if (headDim <= 48)
            {
                if (isUnpacked) shader = MHAAsmShaderWmma::unpacked_128_64x64x48_64x48x64;
                else if (isPackedKV) shader = MHAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64;
                else if (isPackedQK) shader = MHAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64;
                else if (isPackedQKV) shader = MHAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64;
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnpacked) shader = MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64;
                else if (isPackedKV) shader = MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64;
                else if (isPackedQK) shader = MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64;
                else if (isPackedQKV) shader = MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64;
            }
        }

        if (shader == MHAAsmShaderWmma::count)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

        Binaries binaries;

        auto setBlobs = [batchSize, headCount, kvSequenceLength, qSequenceLength, headDim, &binaries, &calcGridAndBlocks]
        <std::uint32_t MPerBlock, std::uint32_t NPerBlock, std::uint32_t BlockSize>
        (const auto& shaderRel, const auto& shaderNonRel, const auto& shaderRelWithStrides, const auto& shaderNonRelWithStrides, 
         const auto& constants, const auto& arg_constants, const auto& arg_constants_with_strides)
         -> void
        {
            Blob relBlob;
            Blob nonRelBlob;
            Blob relBlobWithStrides;
            Blob nonRelBlobWithStrides;

            auto [grid, blocks] = calcGridAndBlocks.template operator()<MPerBlock, NPerBlock, BlockSize>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
            relBlob = std::move(*make_binary_blob(shaderRel));
            relBlob = constants;
            relBlob = arg_constants;
            relBlob.setGridBlocks(grid, blocks);
            nonRelBlob = std::move(*make_binary_blob(shaderNonRel));
            nonRelBlob = constants;
            nonRelBlob = arg_constants;
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

        if(isGfx110x(gfxArch))
        {
            switch (shader)
            {
            case MHAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_self_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_self_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_cross_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_cross_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1100,
                    gfx1100_rel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    gfx1100_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1100,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            default:
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
            }
        }
        else if (isGfx115x(gfxArch))
        {
            switch (shader)
            {
            case MHAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_self_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_self_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_cross_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_cross_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1150,
                    gfx1150_rel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    gfx1150_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1150,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            default:
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
            }
        }
        else if (isGfx120x(gfxArch))
        {
            switch (shader)
            {
            case MHAAsmShaderWmma::unpacked_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_self_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_self_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_cross_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::unpacked_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_unpacked_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::unpacked_q_k_v_cross_attention_ARGS_CONSTANT,
                    fp16::unpacked_q_k_v_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_kv_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_kv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS,
                    fp16::packed_q_kv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qk_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qk_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_qk_ARGS_CONSTANTS,
                    fp16::packed_qk_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x128x80_64x80x64:
                setBlobs.template operator()<64, 80, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x128x80_64x80x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x192x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_128_64x64x48_64x48x64:
                setBlobs.template operator()<64, 48, 128>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
            case MHAAsmShaderWmma::packed_qkv_fallback_64_32x64x48_32x48x64:
                setBlobs.template operator()<32, 48, 64>(
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_no_strides_fp16_gfx1201,
                    gfx1201_rel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    gfx1201_nonrel::multi_head_attention_typed_double_pointer_packed_qkv_fallback_64_32x64x48_32x48x64_forward_with_strides_fp16_gfx1201,
                    fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS,
                    fp16::packed_qkv_self_attention_ARGS_CONSTANTS,
                    fp16::packed_qkv_with_strides_ARGS_CONSTANTS);
                break;
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

} // namespace mlss::shaders::mha::ck::wmma
