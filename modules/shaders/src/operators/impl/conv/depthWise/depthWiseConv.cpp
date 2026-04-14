/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "depthWiseConv.hpp"

namespace mlss::conv::depth_wise
{

    DepthWiseConv::DepthWiseConv(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "DepthWiseConv";
    }

    std::string DepthWiseConv::getCaseName()
    {
        return "AMDMLSS::DepthWiseConv";
    }

    std::expected<Binaries, std::error_code> DepthWiseConv::getBinaries() const
    {
        auto result = BackendSelector<DepthWiseConv>::select(m_attributes, m_gfxIpTriple);
        if (result.binaries.has_value())
        {
            m_implName = result.implName;
        }
        return result.binaries;
    }

    uint32_t DepthWiseConv::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        std::ignore = attributes;
        std::ignore = gfxip;
        std::ignore = context;
        return 0x00000000u;
    }

} // namespace mlss::conv::depth_wise
