/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "ckShadersOp.hpp"
#include "core/core.hpp"
#include "wmma/shadersUtils.hpp"

namespace mlss::mha::ck
{

    //=====================================================================================================================
    // CKMha implementation
    //=====================================================================================================================

    CKMha::CKMha(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "CK-MHA-WMMA";
    }

    std::string CKMha::getOperatorName()
    {
        return "AMDMLSS::CKMha::Wmma";
    }

    std::expected<Binaries, std::error_code> CKMha::getBinaries() const
    {
        return wmma::getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool CKMha::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return wmma::isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::mha::ck
