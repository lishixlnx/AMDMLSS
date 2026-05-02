/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipGemmMxN.hpp"
#include "shadersUtils.hpp"

namespace mlss::gemm::mxn::hip
{

    HipGemmMxN::HipGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HIP-GemmMxN";
    }

    std::string HipGemmMxN::getOperatorName()
    {
        return "AMDMLSS::HipGemmMxN";
    }

    std::expected<Binaries, std::error_code> HipGemmMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool HipGemmMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::gemm::mxn::hip
