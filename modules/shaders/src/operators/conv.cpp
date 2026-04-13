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
        auto* conv1x1 = CaseRegistry<OperatorConv>::get<mlss::conv::one_by_one::Conv1x1>();
        if (conv1x1 != nullptr)
        {
            auto result = conv1x1->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = conv1x1->name;
                return result;
            }
        }

        auto* convMxN = CaseRegistry<OperatorConv>::get<mlss::conv::mxn::ConvMxN>();
        if (convMxN != nullptr)
        {
            auto result = convMxN->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = convMxN->name;
                return result;
            }
        }

        auto* dilated = CaseRegistry<OperatorConv>::get<mlss::conv::dilated::DilatedConv>();
        if (dilated != nullptr)
        {
            auto result = dilated->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = dilated->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    bool OperatorConv::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto params = mlss::conv::utils::buildConvParams(attributes);

        auto* conv1x1 = CaseRegistry<OperatorConv>::get<mlss::conv::one_by_one::Conv1x1>();
        if (conv1x1 != nullptr && conv1x1->getCaps(attributes, gfxip, &params) != 0x00000000u)
        {
            return true;
        }

        auto* convMxN = CaseRegistry<OperatorConv>::get<mlss::conv::mxn::ConvMxN>();
        if (convMxN != nullptr && convMxN->getCaps(attributes, gfxip, &params) != 0x00000000u)
        {
            return true;
        }

        auto* dilated = CaseRegistry<OperatorConv>::get<mlss::conv::dilated::DilatedConv>();
        if (dilated != nullptr && dilated->getCaps(attributes, gfxip, &params) != 0x00000000u)
        {
            return true;
        }

        return false;
    }

} // namespace mlss::op
