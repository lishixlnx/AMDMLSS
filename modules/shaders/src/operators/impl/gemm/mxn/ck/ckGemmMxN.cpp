/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "ckGemmMxN.hpp"
#include "shadersUtils.hpp"

namespace mlss::gemm::mxn::ck
{

    CKGemmMxN::CKGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "CK-GemmMxN";
    }

    std::string CKGemmMxN::getOperatorName()
    {
        return "AMDMLSS::CKGemmMxN";
    }

    std::expected<Binaries, std::error_code> CKGemmMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool CKGemmMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::gemm::mxn::ck
