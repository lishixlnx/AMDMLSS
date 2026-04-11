/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/gqa.hpp"
#include "impl/gqa/ck/ckShadersOp.hpp"

template class mlss::BackendBase<mlss::gqa::ck::CKGqa, mlss::op::OperatorGQA>;

namespace mlss::op
{

    //=====================================================================================================================
    // OperatorGQA implementation
    //=====================================================================================================================

    OperatorGQA::OperatorGQA(const std::vector<Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "GQA";
    }

    std::string OperatorGQA::getOperatorName()
    {
        return "AMDMLSS::OperatorGQA";
    }

    std::expected<Binaries, std::error_code> OperatorGQA::getBinaries() const
    {
        auto result = BackendSelector<OperatorGQA>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    bool OperatorGQA::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return BackendSelector<OperatorGQA>::anyCaps(attributes, gfxArch);
    }

} // namespace mlss::op
