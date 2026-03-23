#include "shadersUtils.hpp"
#include "fp16/shadersConstants.hpp"
#include "fp16/shaderBinGfx1100.hpp"
#include "fp16/shaderBinGfx1150.hpp"
#include "fp16/shaderBinGfx1201.hpp"

namespace mlss::shaders::mha::ck::wmma
{

    namespace 
    {
        // Helper functions to check GFX architecture ranges
        inline bool isGfx110x(GfxArchitectureFlags gfxArch)
        {
            return (gfxArch >= GfxArchitectureFlags::Gfx1100) && 
                   (gfxArch <= GfxArchitectureFlags::Gfx1103);
        }

        inline bool isGfx115x(GfxArchitectureFlags gfxArch)
        {
            return (gfxArch >= GfxArchitectureFlags::Gfx1150) && 
                   (gfxArch <= GfxArchitectureFlags::Gfx1154);
        }

        inline bool isGfx120x(GfxArchitectureFlags gfxArch)
        {
            return (gfxArch >= GfxArchitectureFlags::Gfx1200) && 
                   (gfxArch <= GfxArchitectureFlags::Gfx1201);
        }

        // MHA shader variants for CK implementation
        enum class MHAAsmShaderWmma : std::uint32_t
        {            
            // Unpacked variants (0-4)
            unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16,
            unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16,
            unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16,
            unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16,
            unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16,

    
            // Packed variants - defined as offset from unpacked
            packed_kv_cross_attention_128_64x64x48_64x48x64_forward_fp16,
            packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16,
            packed_qkv_self_attention_128_64x128x80_64x80x64_forward_fp16,
            packed_qkv_self_attention_128_64x192x48_64x48x64_forward_fp16,
            packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16,

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
        if(!isGfx110x(gfxArch) && !isGfx115x(gfxArch) && !isGfx120x(gfxArch))
        {
            return false;
        }

        if (dataType != MLSS_FLOAT16)
        {
            return false;
        }

        // Check for unsupported packing mode (packed QK without V)
        if(packing == 1)  // MLSS_ATTR_CONFIG_GQA_PACKING_PACKED_QK
        {
            return false;
        }

        return (sizeHeads <= 48) || ((sizeHeads % 2) == 0) || ((qSequenceLength == kvSequenceLength) && (sizeHeads <= 80));   
    }

    std::expected<Blob, std::error_code> getWmmaShadersBlob(
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

        MHAAsmShaderWmma shader = MHAAsmShaderWmma::count;

        // Select shader based on attention type and head dimensions
        if (isSelfAttention)
        {
            if (headDim <= 48)
            {
                if (isUnpacked)
                {
                    shader = MHAAsmShaderWmma::unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16;
                }
                else
                {
                    shader = MHAAsmShaderWmma::packed_qkv_self_attention_128_64x192x48_64x48x64_forward_fp16;
                }
            }
            else if (headDim <= 80)
            {
                if (isUnpacked)
                {
                    shader = MHAAsmShaderWmma::unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16;
                }
                else
                {
                    shader = MHAAsmShaderWmma::packed_qkv_self_attention_128_64x128x80_64x80x64_forward_fp16;
                }
            }
            else if ((headDim % 2) == 0)
            {
                if (isUnpacked)
                {
                    shader = MHAAsmShaderWmma::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16;
                }
                else
                {
                    shader = MHAAsmShaderWmma::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16;
                }
            }
            else
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
            }
        }
        else
        {
            if (headDim <= 48)
            {
                if (isUnpacked)
                {
                    shader = MHAAsmShaderWmma::unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16;
                }
                else
                {
                    shader = MHAAsmShaderWmma::packed_kv_cross_attention_128_64x64x48_64x48x64_forward_fp16;
                }
            }
            else if ((headDim % 2) == 0)
            {
                shader = MHAAsmShaderWmma::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16;
            }
            else
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
            }
        }

        Blob blob;
        
        if(isGfx110x(gfxArch))
        {
            switch (shader)
            {
            case MHAAsmShaderWmma::packed_kv_cross_attention_128_64x64x48_64x48x64_forward_fp16:
            {
                blob = std::move(*MLSS_MAKE_BLOB(fp16::gfx1100::packed_kv_cross_attention_128_64x64x48_64x48x64_forward));
                blob = fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_self_attention_128_64x128x80_64x80x64_forward_fp16:
            {
                blob = std::move(*MLSS_MAKE_BLOB(fp16::gfx1100::packed_qkv_self_attention_128_64x128x80_64x80x64_forward));
                blob = fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 80, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_self_attention_128_64x192x48_64x48x64_forward_fp16:
            {
                blob = std::move(*MLSS_MAKE_BLOB(fp16::gfx1100::packed_qkv_self_attention_128_64x192x48_64x48x64_forward));
                blob = fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = std::move(*MLSS_MAKE_BLOB(fp16::gfx1100::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = std::move(*MLSS_MAKE_BLOB(fp16::gfx1100::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1100::unpacked_cross_attention_128_64x64x48_64x48x64_forward));
                blob = fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1100::unpacked_self_attention_128_64x128x80_64x80x64_forward));
                blob = fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 80, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1100::unpacked_self_attention_128_64x192x48_64x48x64_forward));
                blob = fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1100::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1100::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
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
            case MHAAsmShaderWmma::packed_kv_cross_attention_128_64x64x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::packed_kv_cross_attention_128_64x64x48_64x48x64_forward));
                blob = fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_self_attention_128_64x128x80_64x80x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::packed_qkv_self_attention_128_64x128x80_64x80x64_forward));
                blob = fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 80, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_self_attention_128_64x192x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::packed_qkv_self_attention_128_64x192x48_64x48x64_forward));
                blob = fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::unpacked_cross_attention_128_64x64x48_64x48x64_forward));
                blob = fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::unpacked_self_attention_128_64x128x80_64x80x64_forward));
                blob = fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 80, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::unpacked_self_attention_128_64x192x48_64x48x64_forward));
                blob = fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1150::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
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
            case MHAAsmShaderWmma::packed_kv_cross_attention_128_64x64x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::packed_kv_cross_attention_128_64x64x48_64x48x64_forward));
                blob = fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_self_attention_128_64x128x80_64x80x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::packed_qkv_self_attention_128_64x128x80_64x80x64_forward));
                blob = fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 80, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_self_attention_128_64x192x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::packed_qkv_self_attention_128_64x192x48_64x48x64_forward));
                blob = fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::unpacked_cross_attention_128_64x64x48_64x48x64_forward));
                blob = fp16::cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::unpacked_self_attention_128_64x128x80_64x80x64_forward));
                blob = fp16::self_attention_128_64x128x80_64x80x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 80, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::unpacked_self_attention_128_64x192x48_64x48x64_forward));
                blob = fp16::self_attention_128_64x192x48_64x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<64, 48, 128>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_q_kv_cross_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
                break;
            }
            case MHAAsmShaderWmma::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16:
            {
                blob = (*MLSS_MAKE_BLOB(fp16::gfx1201::unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward));
                blob = fp16::fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS;
                blob = fp16::packed_qkv_self_attention_ARGS_CONSTANTS;
                auto [grid, blocks] = calcGridAndBlocks.template operator()<32, 48, 64>(batchSize, headCount, kvSequenceLength, qSequenceLength, headDim);
                blob.setGridBlocks(grid, blocks);
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

        return blob;
    }

} // namespace mlss::shaders::mha::ck::wmma
