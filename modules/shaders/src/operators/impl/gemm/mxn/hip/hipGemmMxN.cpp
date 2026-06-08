/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipGemmMxN.hpp"
#include "shadersUtils.hpp"
#include "shaders/src/operators/impl/gemm/utils.hpp"

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
        const auto params = mlss::gemm::utils::buildGemmParams(m_attributes);
        return getShadersBlob(m_gfxIpTriple, m_attributes, &params);
    }

    uint32_t HipGemmMxN::getCapsImpl(const std::vector<Attribute>& attributes,
                                     const GfxIpTriple& gfxArch,
                                     const void* context)
    {
        return isShadersAvailable(gfxArch, attributes, context).values;
    }

} // namespace mlss::gemm::mxn::hip
