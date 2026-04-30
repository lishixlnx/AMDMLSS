/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "winogradRageShadersOp.hpp"
#include "../../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::rage
{

    WinogradRageConvMxN::WinogradRageConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base_t(attributes, gfxip)
    {
        this->m_implName = "Winograd-Rage-ConvMxN";
    }

    std::string WinogradRageConvMxN::getOperatorName()
    {
        return "AMDMLSS::WinogradRageConvMxN";
    }

    std::expected<Binaries, std::error_code> WinogradRageConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, mlss::conv::utils::buildConvParams(m_attributes));
    }

    uint32_t WinogradRageConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
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

} // namespace mlss::conv::mxn::winograd::rage
