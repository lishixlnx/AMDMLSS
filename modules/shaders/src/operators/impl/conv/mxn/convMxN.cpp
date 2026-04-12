/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "convMxN.hpp"

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
        std::ignore = attributes;
        std::ignore = gfxip;
        std::ignore = context;
        return 0x00000000u;
    }

} // namespace mlss::conv::mxn
