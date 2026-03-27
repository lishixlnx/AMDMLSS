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

    std::expected<Binaries, std::error_code> OperatorMHA::getBinaries() const
    {
        if (ck::CKMha::getCapsImpl(m_attributes, m_gfxArch))
        {
            ck::CKMha ckMha(m_attributes, m_gfxArch);
            m_implName = ckMha.getOperatorName();
            return ckMha.getBinaries();
        }
        else
        {
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
    }

    bool OperatorMHA::getCapsImpl(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxArch)
    {
        return ck::CKMha::getCapsImpl(attributes, gfxArch);
    }

} // namespace mlss::shaders::op
