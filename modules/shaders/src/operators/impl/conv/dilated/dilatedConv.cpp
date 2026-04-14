/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "dilatedConv.hpp"
#include "../utils.hpp"
#include "hip/hipShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::dilated::hip::HipDilatedConv, mlss::conv::dilated::DilatedConv>;

using namespace mlss::conv::utils;
using mlss::op::utils::MetaCmdCaps;

namespace mlss::conv::dilated
{

    DilatedConv::DilatedConv(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "DilatedConv";
    }

    std::string DilatedConv::getCaseName()
    {
        return "AMDMLSS::DilatedConv";
    }

    std::expected<Binaries, std::error_code> DilatedConv::getBinaries() const
    {
        if (auto* hipEntry = BackendRegistry<DilatedConv>::get<hip::HipDilatedConv>(); hipEntry != nullptr)
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

    uint32_t DilatedConv::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        GenericConvParams params{};
        MetaCmdCaps caps{.values = 0x00000000u};

        if (context != nullptr)
        {
            params = *static_cast<const GenericConvParams*>(context);
        }
        else if (!attributes.empty())
        {
            params = buildConvParams(attributes);
        }

        if (auto* hipEntry = BackendRegistry<DilatedConv>::get<hip::HipDilatedConv>(); hipEntry != nullptr)
        {
            caps.values = hipEntry->getCaps(attributes, gfxip, &params);
        }

        return caps.values;
    }

} // namespace mlss::conv::dilated
