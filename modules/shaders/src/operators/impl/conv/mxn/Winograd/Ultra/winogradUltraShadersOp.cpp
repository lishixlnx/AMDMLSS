/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "winogradUltraShadersOp.hpp"
#include "../../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::mxn::winograd::ultra
{

#if 0

    WinogradUltraConvMxN::WinogradUltraConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "Winograd-Ultra-ConvMxN";
    }

    std::string WinogradUltraConvMxN::getOperatorName()
    {
        return "AMDMLSS::WinogradUltraConvMxN";
    }

    std::expected<Binaries, std::error_code> WinogradUltraConvMxN::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, mlss::conv::utils::buildConvParams(m_attributes));
    }

    uint32_t WinogradUltraConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
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

    #else

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        std::ignore = gfxip;
        std::ignore = params;

        return false;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params)
    {
        std::ignore = gfxip;
        std::ignore = params;

        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedOperator));
    }


    #endif

} // namespace mlss::conv::mxn::winograd::ultra
