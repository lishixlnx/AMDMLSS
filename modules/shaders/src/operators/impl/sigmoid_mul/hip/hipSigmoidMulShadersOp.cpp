/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipSigmoidMulShadersOp.hpp"
#include "shadersUtils.hpp"

namespace mlss::sigmoid_mul::hip
{

    HipSigmoidMul::HipSigmoidMul(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HIP-SigmoidMul";
    }

    std::string HipSigmoidMul::getOperatorName()
    {
        return "AMDMLSS::HipSigmoidMul";
    }

    std::expected<Binaries, std::error_code> HipSigmoidMul::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool HipSigmoidMul::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::sigmoid_mul::hip
