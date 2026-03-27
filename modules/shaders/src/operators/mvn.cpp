/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/mvn.hpp"

namespace mlss::shaders::op
{

    //=====================================================================================================================
    // OperatorMVN implementation
    //=====================================================================================================================

    OperatorMVN::OperatorMVN(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
        : base(attributes, gfxip)
    {
    }

    std::string OperatorMVN::getOperatorName()
    {
        return "AMDMLSS::OperatorMVN";
    }

    std::expected<Binaries, std::error_code> OperatorMVN::getBinaries() const
    {
        return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
    }

    bool OperatorMVN::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
    {
        return false;
    }

} // namespace mlss::shaders::op
