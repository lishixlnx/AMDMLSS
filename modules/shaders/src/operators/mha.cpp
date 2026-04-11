/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/mha.hpp"
#include "impl/mha/ck/ckShadersOp.hpp"

template class mlss::BackendBase<mlss::mha::ck::CKMha, mlss::op::OperatorMHA>;

namespace mlss::op
{

    //=====================================================================================================================
    // OperatorMHA implementation
    //=====================================================================================================================

    OperatorMHA::OperatorMHA(const std::vector<Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "MHA";
    }

    std::string OperatorMHA::getOperatorName()
    {
        return "AMDMLSS::OperatorMHA";
    }

    std::expected<Binaries, std::error_code> OperatorMHA::getBinaries() const
    {
        auto result = BackendSelector<OperatorMHA>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    bool OperatorMHA::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return BackendSelector<OperatorMHA>::anyCaps(attributes, gfxArch);
    }

} // namespace mlss::op
