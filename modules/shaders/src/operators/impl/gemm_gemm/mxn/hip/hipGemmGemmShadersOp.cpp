/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipGemmGemmShadersOp.hpp"
#include "shadersUtils.hpp"

namespace mlss::gemm_gemm::mxn::hip
{

    HipGemmGemmMxN::HipGemmGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HIP-GemmGemm-MxN";
    }

    std::string HipGemmGemmMxN::getOperatorName()
    {
        return "AMDMLSS::HipGemmGemmMxN";
    }

    std::expected<Binaries, std::error_code> HipGemmGemmMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool HipGemmGemmMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::gemm_gemm::mxn::hip
