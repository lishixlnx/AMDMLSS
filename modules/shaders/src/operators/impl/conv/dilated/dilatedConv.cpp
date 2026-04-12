/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "dilatedConv.hpp"
#include "hip/hipShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::dilated::hip::HipDilatedConv, mlss::conv::dilated::DilatedConv>;

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
        auto result = BackendSelector<DilatedConv>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    uint32_t DilatedConv::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        return BackendSelector<DilatedConv>::bestCaps(attributes, gfxip, context);
    }

} // namespace mlss::conv::dilated
