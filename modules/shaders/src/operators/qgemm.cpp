/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/qgemm.hpp"

namespace mlss::op
{

    //=====================================================================================================================
    // OperatorGEMM implementation
    //=====================================================================================================================

    OperatorQGEMM::OperatorQGEMM(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
    }

    std::string OperatorQGEMM::getOperatorName()
    {
        return "AMDMLSS::OperatorQGEMM";
    }

    std::expected<Binaries, std::error_code> OperatorQGEMM::getBinaries() const
    {
        return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
    }

    bool OperatorQGEMM::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
    {
        return false;
    }

} // namespace mlss::op
