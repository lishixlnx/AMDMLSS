/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "hipConv1x1ShadersOp.hpp"
#include "../../../utils.hpp"
#include "wmma/shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::one_by_one::hip::wmma
{

    HipConv1x1::HipConv1x1(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HipWmma-Conv1x1";
    }

    std::string HipConv1x1::getOperatorName()
    {
        return "AMDMLSS::HipConv1x1";
    }

    std::expected<Binaries, std::error_code> HipConv1x1::getBinaries() const
    {
        return getShadersBlob(m_gfxIpTriple, mlss::conv::utils::buildConvParams(m_attributes));
    }

    uint32_t HipConv1x1::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
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

} // namespace mlss::conv::one_by_one::hip::wmma
