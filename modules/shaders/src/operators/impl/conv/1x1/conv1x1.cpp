/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "conv1x1.hpp"
#include "misa/misaShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::one_by_one::misa::MisaConv1x1, mlss::conv::one_by_one::Conv1x1>;

namespace mlss::conv::one_by_one
{

    Conv1x1::Conv1x1(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "Conv1x1";
    }

    std::string Conv1x1::getCaseName()
    {
        return "AMDMLSS::Conv1x1";
    }

    std::expected<Binaries, std::error_code> Conv1x1::getBinaries() const
    {
        auto result = BackendSelector<Conv1x1>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    uint32_t Conv1x1::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        return BackendSelector<Conv1x1>::bestCaps(attributes, gfxip, context);
    }

} // namespace mlss::conv::one_by_one
