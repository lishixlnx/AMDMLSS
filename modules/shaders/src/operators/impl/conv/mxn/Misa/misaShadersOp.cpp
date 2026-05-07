/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "misaShadersOp.hpp"
#include "../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::misa
{

    MisaConvMxN::MisaConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "MISA-ConvMxN";
    }

    std::string MisaConvMxN::getOperatorName()
    {
        return "AMDMLSS::MisaConvMxN";
    }

    std::expected<Binaries, std::error_code> MisaConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, m_attributes, nullptr);
    }

    uint32_t MisaConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
    {
        return isShadersAvailable(gfxArch, attributes, context).values;
    }

} // namespace mlss::conv::mxn::misa
