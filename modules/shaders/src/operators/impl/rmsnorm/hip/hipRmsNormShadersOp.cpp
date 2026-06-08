/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipRmsNormShadersOp.hpp"
#include "shadersUtils.hpp"

namespace mlss::rmsnorm::hip
{

    HipRmsNorm::HipRmsNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HIP-RmsNorm";
    }

    std::string HipRmsNorm::getOperatorName()
    {
        return "AMDMLSS::HipRmsNorm";
    }

    std::expected<Binaries, std::error_code> HipRmsNorm::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool HipRmsNorm::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::rmsnorm::hip
