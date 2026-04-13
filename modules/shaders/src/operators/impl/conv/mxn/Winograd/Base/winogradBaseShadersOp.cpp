/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "winogradBaseShadersOp.hpp"
#include "../../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::base
{

    WinogradBaseConvMxN::WinogradBaseConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base_t(attributes, gfxip)
    {
        this->m_implName = "Winograd-Base-ConvMxN";
    }

    std::string WinogradBaseConvMxN::getOperatorName()
    {
        return "AMDMLSS::WinogradBaseConvMxN";
    }

    std::expected<Binaries, std::error_code> WinogradBaseConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, mlss::conv::utils::buildConvParams(m_attributes));
    }

    uint32_t WinogradBaseConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
    {
        mlss::conv::utils::GenericConvParams params{};

        if (context != nullptr)
        {
            params = *static_cast<const mlss::conv::utils::GenericConvParams*>(context);
        }
        else if (!attributes.empty())
        {
            params = mlss::conv::utils::buildConvParams(attributes);
        }

        return isShadersAvailable(gfxArch, params).values;
    }

} // namespace mlss::conv::mxn::winograd::base
