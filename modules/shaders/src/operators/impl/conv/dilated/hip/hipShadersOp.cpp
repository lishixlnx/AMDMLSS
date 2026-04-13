/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipShadersOp.hpp"
#include "../../utils.hpp"
#include "wmma/shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::dilated::hip
{

    HipDilatedConv::HipDilatedConv(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HIP-DilatedConv-WMMA";
    }

    std::string HipDilatedConv::getOperatorName()
    {
        return "AMDMLSS::HipDilatedConv::Wmma";
    }

    std::expected<Binaries, std::error_code> HipDilatedConv::getBinaries() const
    {
        return wmma::getShadersBlob(m_gfxIpTriple, mlss::conv::utils::buildConvParams(m_attributes));
    }

    uint32_t HipDilatedConv::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
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

        return wmma::isShadersAvailable(gfxArch, params).values;
    }

} // namespace mlss::conv::dilated::hip
