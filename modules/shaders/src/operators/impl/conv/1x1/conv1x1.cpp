/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "conv1x1.hpp"
#include "misa/misaShadersOp.hpp"
#include "hip/hipConv1x1ShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::one_by_one::misa::MisaConv1x1, mlss::conv::one_by_one::Conv1x1>;
template class mlss::BackendBase<mlss::conv::one_by_one::hip::wmma::HipConv1x1, mlss::conv::one_by_one::Conv1x1>;

using namespace mlss::conv::utils;

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
        GenericConvParams params{};

        Binaries binaries;        

        auto result = BackendSelector<Conv1x1>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }

        if(context != nullptr)
        {
            auto params = *static_cast<const mlss::conv::utils::GenericConvParams*>(context);
        }
        else if(!attributes.empty())
        {
            params = mlss::conv::utils::buildConvParams(attributes);
        }

        if(params.r == 1 && params.s == 1)
        {
            if(auto* misaEntry = BackendRegistry<Conv1x1>::get<misa::MisaConv1x1>(); misaEntry != nullptr)
            {
                if(tmp = misaEntry->getBinaries(attributes, gfxip), tmp.has_value())
                {
                    binaries = tmp.value();
                }
                else
                {
                    return std::unexpected(tmp.error());
                }
            }
            else if (auto* hipEntry = BackendRegistry<Conv1x1>::get<hip::wmma::HipConv1x1>(); hipEntry != nullptr)
            {
                if(tmp = hipEntry->getBinaries(attributes, gfxip), tmp.has_value())
                {
                    binaries = tmp.value();
                }
                else
                {
                    return std::unexpected(tmp.error());
                }
            }
        }
        else
        {
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }

        return binaries;
    }

    uint32_t Conv1x1::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        GenericConvParams params{};

        MetaCmdCaps caps{.values = 0x00000000u};

        if(context != nullptr)
        {
            auto params = *static_cast<const mlss::conv::utils::GenericConvParams*>(context);
        }
        else if(!attributes.empty())
        {
            params = mlss::conv::utils::buildConvParams(attributes);
        }

        if(params.r == 1 && params.s == 1)
        {
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
