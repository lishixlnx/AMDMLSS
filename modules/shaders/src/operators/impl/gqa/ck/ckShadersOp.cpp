/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "ckShadersOp.hpp"
#include "core/core.hpp"
#include "wmma/shadersUtils.hpp"

namespace mlss::gqa::ck
{

    //=====================================================================================================================
    // CKGqa implementation
    //=====================================================================================================================

    CKGqa::CKGqa(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "CK-GQA-WMMA";
    }

    std::string CKGqa::getOperatorName()
    {
        return "AMDMLSS::CKGqa::Wmma";
    }

    std::expected<Binaries, std::error_code> CKGqa::getBinaries() const
    {
        return wmma::getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    bool CKGqa::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch)
    {
        return wmma::isShadersAvailable(gfxArch, attributes, nullptr);
    }

} // namespace mlss::gqa::ck
