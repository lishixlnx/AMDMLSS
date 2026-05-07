/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipMvn2ShadersOp.hpp"
#include "shadersUtils.hpp"

namespace mlss::mvn::mvn2::hip
{

    HipMvn2InstaNorm::HipMvn2InstaNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HIP-MVN2-InstaNorm";
    }

    std::string HipMvn2InstaNorm::getOperatorName()
    {
        return "AMDMLSS::HipMvn2InstaNorm";
    }

    std::expected<Binaries, std::error_code> HipMvn2InstaNorm::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool HipMvn2InstaNorm::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::mvn::mvn2::hip
