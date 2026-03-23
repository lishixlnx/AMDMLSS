/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "shaders/operators/mha.hpp"
#include "impl/mha/ck/ckShadersOp.hpp"

namespace mlss::shaders::op
{

    using namespace mlss::shaders::mha;

        //=====================================================================================================================
        // OperatorMHA implementation
        //=====================================================================================================================

        OperatorMHA::OperatorMHA(const std::vector<Attribute>& attributes, GfxArchitectureFlags gfxip)
            : base(attributes, gfxip)
        {
            this->m_implName = "HipMHA";
        }

        std::string OperatorMHA::getOperatorName()
        {
            return "AMDMLSS::OperatorMHA";
        }

        std::expected<OperatorMHA::blob, std::error_code> OperatorMHA::getBlob() const
        {
            if(ck::CKMha::getCapsImpl(m_attributes, m_gfxArch))
            {
                ck::CKMha ckMha(m_attributes, m_gfxArch);
                m_implName = ckMha.getOperatorName();
                return ckMha.getBlob();
            }
            else
            {
                return std::unexpected(std::make_error_code(std::errc::invalid_argument));
            }
        }

        bool OperatorMHA::getCapsImpl(const std::vector<Attribute>& attributes)
        {
            // Extract the GFX architecture from the current instance
            // Note: This is a static method, so we can't access instance members
            // The architecture check will be done elsewhere
            
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
            std::array<std::uint32_t, 4> q_strides;
            std::array<std::uint32_t, 4> k_strides;
            std::array<std::uint32_t, 4> v_strides;
            std::array<std::uint32_t, 4> output_strides;

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
                else if (attribute.is(MLSS_ATTR_MHA_QSTRIDES))
                {
                    q_strides = attribute.value<std::array<std::uint32_t, 4>>();
                }
                else if (attribute.is(MLSS_ATTR_MHA_KSTRIDES))
                {
                    k_strides = attribute.value<std::array<std::uint32_t, 4>>();
                }
                else if (attribute.is(MLSS_ATTR_MHA_VSTRIDES))
                {
                    v_strides = attribute.value<std::array<std::uint32_t, 4>>();
                }
                else if (attribute.is(MLSS_ATTR_MHA_OUTPUTSTRIDES))
                {
                    output_strides = attribute.value<std::array<std::uint32_t, 4>>();
                }
            }

            // Check required parameters
            if ((batch == 0) || (q_seq == 0) || (kv_seq == 0) || (size_heads == 0) ||
                (head_count == 0) || (scale == 0) || (data_type != MLSS_FLOAT16))
            {
                return false;
            }

            bool isSupported = true;

            // Check packing values
            if (!((packing == 0) || (packing == 2) || (packing == 3)))
            {
                isSupported = false;
            }

            // Check data type
            if (data_type != MLSS_FLOAT16)
            {
                isSupported = false;
            }

            // Check cross-attention specific constraints
            bool isCrossAttention = q_seq != kv_seq;
            if (isCrossAttention && (size_heads > 80) && ((size_heads % 2) != 0))
            {
                isSupported = false;
            }
            else if (isCrossAttention && (size_heads > 48) && ((size_heads % 2) != 0))
            {
                isSupported = false;
            }

            // Note: GFX architecture check (isGfx11Plus) should be done by the caller
            // since this is a static method and doesn't have access to instance data
            
            return isSupported;
        }

} // namespace mlss::shaders::op
