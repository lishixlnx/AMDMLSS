/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "misaShadersOp.hpp"
#include "../../utils.hpp"
#include "shadersUtils.hpp"
#include "core/core.hpp"

namespace mlss::conv::one_by_one::misa
{

    MisaConv1x1::MisaConv1x1(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "MISA-Conv1x1";
    }

    std::string MisaConv1x1::getOperatorName()
    {
        return "AMDMLSS::MisaConv1x1";
    }

    std::expected<Binaries, std::error_code> MisaConv1x1::getBinaries() const
    {
        return getMisaShadersBlob(m_gfxIpTriple, mlss::conv::utils::buildConvParams(m_attributes));
    }

    uint32_t MisaConv1x1::getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context)
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

        return isMisaShadersAvailable(gfxArch, params).values;
    }

} // namespace mlss::conv::one_by_one::misa
