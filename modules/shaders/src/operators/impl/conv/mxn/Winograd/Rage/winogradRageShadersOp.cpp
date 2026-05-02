/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "winogradRageShadersOp.hpp"
#include "../../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::rage
{

    WinogradRageConvMxN::WinogradRageConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base_t(attributes, gfxip)
    {
        this->m_implName = "Winograd-Rage-ConvMxN";
    }

    std::string WinogradRageConvMxN::getOperatorName()
    {
        return "AMDMLSS::WinogradRageConvMxN";
    }

    std::expected<Binaries, std::error_code> WinogradRageConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    uint32_t WinogradRageConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
    {
        return isShadersAvailable(gfxArch, attributes, context).values;
    }

} // namespace mlss::conv::mxn::winograd::rage
