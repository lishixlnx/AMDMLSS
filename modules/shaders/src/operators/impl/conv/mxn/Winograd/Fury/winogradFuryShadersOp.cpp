/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "winogradFuryShadersOp.hpp"
#include "../../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::fury
{

    WinogradFuryConvMxN::WinogradFuryConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base_t(attributes, gfxip)
    {
        this->m_implName = "Winograd-Fury-ConvMxN";
    }

    std::string WinogradFuryConvMxN::getOperatorName()
    {
        return "AMDMLSS::WinogradFuryConvMxN";
    }

    std::expected<Binaries, std::error_code> WinogradFuryConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    uint32_t WinogradFuryConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
    {
        return isShadersAvailable(gfxArch, attributes, context).values;
    }

} // namespace mlss::conv::mxn::winograd::fury
