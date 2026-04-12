/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/conv.hpp"
#include "impl/conv/utils.hpp"
#include "impl/conv/mxn/convMxN.hpp"
#include "impl/conv/1x1/conv1x1.hpp"
#include "impl/conv/dilated/dilatedConv.hpp"
#include "impl/conv/depthWise/depthWiseConv.hpp"

template class mlss::CaseBase<mlss::conv::mxn::ConvMxN, mlss::op::OperatorConv>;
template class mlss::CaseBase<mlss::conv::one_by_one::Conv1x1, mlss::op::OperatorConv>;
template class mlss::CaseBase<mlss::conv::dilated::DilatedConv, mlss::op::OperatorConv>;
template class mlss::CaseBase<mlss::conv::depth_wise::DepthWiseConv, mlss::op::OperatorConv>;

namespace mlss::op
{

    OperatorConv::OperatorConv(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "Conv";
    }

    std::string OperatorConv::getOperatorName()
    {
        return "AMDMLSS::OperatorConv";
    }

    std::expected<Binaries, std::error_code> OperatorConv::getBinaries() const
    {
        auto result = CaseSelector<OperatorConv>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    bool OperatorConv::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto params = mlss::conv::utils::buildConvParams(attributes);
        return CaseSelector<OperatorConv>::anyCaps(attributes, gfxip, &params);
    }

} // namespace mlss::op
