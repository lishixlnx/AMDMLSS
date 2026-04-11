/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "ckShadersOp.hpp"
#include "core/core.hpp"
#include "wmma/shadersUtils.hpp"

namespace mlss::mha::ck
{

    //=====================================================================================================================
    // CKMha implementation
    //=====================================================================================================================

    CKMha::CKMha(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "CK-MHA-WMMA";
    }

    std::string CKMha::getOperatorName()
    {
        return "AMDMLSS::CKMha::Wmma";
    }

    std::expected<Binaries, std::error_code> CKMha::getBinaries() const
    {
        std::uint32_t batchSize{0};
        std::uint32_t headCount{0};
        std::uint32_t sizeHeads{0};
        std::uint32_t kvSequenceLength{0};
        std::uint32_t qSequenceLength{0};
        std::uint32_t packing{0};
        std::uint32_t dataType{0};

        for (const auto& attribute : m_attributes)
        {
            if (attribute.is(MLSS_ATTR_MHA_BATCH))
            {
                batchSize = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_HEADCOUNT))
            {
                headCount = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_SIZEHEADS))
            {
                sizeHeads = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_KVSEQ))
            {
                kvSequenceLength = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_QSEQ))
            {
                qSequenceLength = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_PACKING))
            {
                packing = attribute.template value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_DATATYPE))
            {
                dataType = attribute.template value<std::uint32_t>();
            }
        }

        return wmma::getWmmaShadersBlob(m_gfxIpTriple, batchSize, headCount, sizeHeads, kvSequenceLength, qSequenceLength, packing, dataType);
    }

    bool CKMha::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        // Extract MHA parameters from attributes
        std::uint32_t batch = 0;
        std::uint32_t q_seq = 0;
        std::uint32_t kv_seq = 0;
        std::uint32_t k_dim = 0;
        std::uint32_t v_dim = 0;
        std::uint32_t size_heads = 0;
        std::uint32_t packing = 0;
        std::uint32_t head_count = 0;
        float scale = 0.f;
        std::uint32_t data_type = 0;

        for (const auto& attribute : attributes)
        {
            if (attribute.is(MLSS_ATTR_MHA_BATCH))
            {
                batch = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_QSEQ))
            {
                q_seq = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_KVSEQ))
            {
                kv_seq = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_KDIM))
            {
                k_dim = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_VDIM))
            {
                v_dim = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_SIZEHEADS))
            {
                size_heads = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_PACKING))
            {
                packing = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_HEADCOUNT))
            {
                head_count = attribute.value<std::uint32_t>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_SCALE))
            {
                scale = attribute.value<float>();
            }
            else if (attribute.is(MLSS_ATTR_MHA_DATATYPE))
            {
                data_type = attribute.value<std::uint32_t>();
            }
        }

        // Check required parameters
        if ((batch == 0) || (q_seq == 0) || (kv_seq == 0) || (size_heads == 0) ||
            (head_count == 0) || (scale == 0.f) || (data_type != MLSS_FLOAT16))
        {
            return false;
        }

        return wmma::isWmmaShadersAvailable(gfxArch, size_heads, q_seq, kv_seq, packing, data_type);
    }

} // namespace mlss::mha::ck
