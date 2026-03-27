/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/dconv.hpp"

namespace mlss::shaders::op
{

    //=====================================================================================================================
    // OperatorDepthWiseConv implementation
    //=====================================================================================================================

    OperatorDepthWiseConv::OperatorDepthWiseConv(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
        : base(attributes, gfxip)
    {
    }

    std::string OperatorDepthWiseConv::getOperatorName()
    {
        return "AMDMLSS::OperatorDepthWiseConv";
    }

    std::expected<Binaries, std::error_code> OperatorDepthWiseConv::getBinaries() const
    {
        return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
    }

    bool OperatorDepthWiseConv::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
    {
        return false;
    }

    //=====================================================================================================================
    // OperatorDilatedConv implementation
    //=====================================================================================================================

    OperatorDilatedConv::OperatorDilatedConv(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
        : base(attributes, gfxip)
    {
    }

    std::string OperatorDilatedConv::getOperatorName()
    {
        return "AMDMLSS::OperatorDilatedConv";
    }

    std::expected<Binaries, std::error_code> OperatorDilatedConv::getBinaries() const
    {
        return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
    }

    bool OperatorDilatedConv::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
    {
        return false;
    }

} // namespace mlss::shaders::op
