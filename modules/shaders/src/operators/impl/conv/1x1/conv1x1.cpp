/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "conv1x1.hpp"
#include "shaders/src/operators/impl/conv/utils.hpp"
#include "misa/misaShadersOp.hpp"
#include "hip/hipConv1x1ShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::one_by_one::misa::MisaConv1x1, mlss::conv::one_by_one::Conv1x1>;
template class mlss::BackendBase<mlss::conv::one_by_one::hip::wmma::HipConv1x1, mlss::conv::one_by_one::Conv1x1>;

using namespace mlss::conv::utils;
using mlss::op::utils::MetaCmdCaps;

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
        if (auto* misaEntry = BackendRegistry<Conv1x1>::get<misa::MisaConv1x1>(); misaEntry != nullptr)
        {
            auto result = misaEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = misaEntry->name;
                return result;
            }
        }

        if (auto* hipEntry = BackendRegistry<Conv1x1>::get<hip::wmma::HipConv1x1>(); hipEntry != nullptr)
        {
            auto result = hipEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = hipEntry->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    uint32_t Conv1x1::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        GenericConvParams params{};

        MetaCmdCaps caps{.values = 0x00000000u};

        if(context != nullptr)
        {
            params = *static_cast<const GenericConvParams*>(context);
        }
        else if(!attributes.empty())
        {
            params = buildConvParams(attributes);
        }

        if(params.r == 1 && params.s == 1)
        {
            if (params.dilationX > 1 || params.dilationY > 1)
            {
                return caps.values;
            }
            if(auto* misaEntry = BackendRegistry<Conv1x1>::get<misa::MisaConv1x1>(); misaEntry != nullptr)
            {
                caps.values = misaEntry->getCaps(attributes, gfxip, &params);
            }
            else if (auto* hipEntry = BackendRegistry<Conv1x1>::get<hip::wmma::HipConv1x1>(); hipEntry != nullptr)
            {
                caps.values = hipEntry->getCaps(attributes, gfxip, &params);
            }            
        }

        return caps.values;
    }

} // namespace mlss::conv::one_by_one
