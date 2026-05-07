/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "winogradBaseShadersOp.hpp"
#include "../../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::base
{

    WinogradBaseConvMxN::WinogradBaseConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base_t(attributes, gfxip)
    {
        this->m_implName = "Winograd-Base-ConvMxN";
    }

    std::string WinogradBaseConvMxN::getOperatorName()
    {
        return "AMDMLSS::WinogradBaseConvMxN";
    }

    std::expected<Binaries, std::error_code> WinogradBaseConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    uint32_t WinogradBaseConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
    {
        return isShadersAvailable(gfxArch, attributes, context).values;
    }

} // namespace mlss::conv::mxn::winograd::base
