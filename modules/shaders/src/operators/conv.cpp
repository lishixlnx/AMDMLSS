/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/conv.hpp"

namespace mlss::shaders::op
{

    //=====================================================================================================================
    // OperatorConv implementation
    //=====================================================================================================================

    OperatorConv::OperatorConv(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "HipConv";
    }

    std::string OperatorConv::getOperatorName()
    {
        return "AMDMLSS::OperatorConv";
    }

    std::expected<Binaries, std::error_code> OperatorConv::getBinaries() const
    {
        return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
    }

    bool OperatorConv::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
    {
        return false;
    }

} // namespace mlss::shaders::op
