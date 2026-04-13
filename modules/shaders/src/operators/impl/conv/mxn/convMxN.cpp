/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "convMxN.hpp"
#include "Misa/misaShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::mxn::misa::MisaConvMxN, mlss::conv::mxn::ConvMxN>;

namespace mlss::conv::mxn
{

    ConvMxN::ConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "ConvMxN";
    }

    std::string ConvMxN::getCaseName()
    {
        return "AMDMLSS::ConvMxN";
    }

    std::expected<Binaries, std::error_code> ConvMxN::getBinaries() const
    {
        auto result = BackendSelector<ConvMxN>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    uint32_t ConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        return BackendSelector<ConvMxN>::bestCaps(attributes, gfxip, context);
    }

} // namespace mlss::conv::mxn
