/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "ckShadersOp.hpp"
#include "core/core.hpp"
#include "wmma/shadersUtils.hpp"

namespace mlss::shaders::gqa::ck
{

    //=====================================================================================================================
    // CKGqa implementation
    //=====================================================================================================================

    CKGqa::CKGqa(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "CK-GQA-WMMA";
    }

    std::string CKGqa::getOperatorName()
    {
        return "AMDMLSS::CKGqa::Wmma";
    }

    std::expected<Binaries, std::error_code> CKGqa::getBinaries() const
    {
        // Extract GQA parameters from attributes
        std::uint32_t batchSize{ 1 };
        std::uint32_t qHeadCount{ 1 };
        std::uint32_t kvHeadCount{ 1 };
        std::uint32_t sizeHeads{ 0 };
        std::uint32_t kvSequenceLength{ 0 };
        std::uint32_t qSequenceLength{ 0 };
        std::uint32_t dataType{ 0 };
        std::uint32_t packing{ MLSS_ATTR_CONFIG_GQA_PACKING_UNPACKED };
        bool useStrides = false;

        for (const auto& attribute : m_attributes)
        {
            if (attribute.is(MLSS_ATTR_GQA_BATCH))
            {
                batchSize = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_QHEADCOUNT))
            {
                qHeadCount = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_KVHEADCOUNT))
            {
                kvHeadCount = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_SIZEHEADS))
            {
                sizeHeads = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_KVSEQ))
            {
                kvSequenceLength = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_QSEQ))
            {
                qSequenceLength = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_DATATYPE))
            {
                dataType = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_PACKING))
            {
                packing = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_QSTRIDES) ||
                     attribute.is(MLSS_ATTR_GQA_KSTRIDES) ||
                     attribute.is(MLSS_ATTR_GQA_VSTRIDES) ||
                     attribute.is(MLSS_ATTR_GQA_OUTPUTSTRIDES))
            {
                useStrides = true;
            }
        }

        return wmma::getWmmaShadersBlob(
            m_gfxArch,
            batchSize,
            qHeadCount,
            kvHeadCount,
            sizeHeads,
            kvSequenceLength,
            qSequenceLength,
            packing,
            useStrides,
            dataType);
    }

    bool CKGqa::getCapsImpl(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxArch)
    {
        // Extract GQA parameters from attributes
        std::uint32_t batch = 0;
        std::uint32_t q_seq = 0;
        std::uint32_t kv_seq = 0;
        std::uint32_t size_heads = 0;
        std::uint32_t q_head_count = 0;
        std::uint32_t kv_head_count = 0;
        float scale = 0.f;
        std::uint32_t data_type = 0;

        for (const auto& attribute : attributes)
        {
            if (attribute.is(MLSS_ATTR_GQA_BATCH))
            {
                batch = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_QSEQ))
            {
                q_seq = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_KVSEQ))
            {
                kv_seq = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_SIZEHEADS))
            {
                size_heads = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_QHEADCOUNT))
            {
                q_head_count = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_KVHEADCOUNT))
            {
                kv_head_count = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_SCALE))
            {
                scale = attribute.value<float>();
            }
            else if (attribute.is(MLSS_ATTR_GQA_DATATYPE))
            {
                data_type = attribute.value<std::uint32_t>();
            }
        }

        // Check required parameters
        if ((batch == 0) || (q_seq == 0) || (kv_seq == 0) || (size_heads == 0) ||
            (q_head_count == 0) || (kv_head_count == 0) || (scale == 0.f) || (data_type != MLSS_FLOAT16))
        {
            return false;
        }

        return wmma::isWmmaShadersAvailable(gfxArch, size_heads, kv_seq, q_seq, data_type);
    }

} // namespace mlss::shaders::gqa::ck

