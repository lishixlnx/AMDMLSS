/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/gemm.hpp"

namespace mlss::shaders::op
{

    //=====================================================================================================================
    // OperatorGEMM implementation
    //=====================================================================================================================

    OperatorGEMM::OperatorGEMM(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
        : base(attributes, gfxip)
    {
    }

    std::string OperatorGEMM::getOperatorName()
    {
        return "AMDMLSS::OperatorGEMM";
    }

    std::expected<Binaries, std::error_code> OperatorGEMM::getBinaries() const
    {
        return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
    }

    bool OperatorGEMM::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
    {
        return false;
    }

} // namespace mlss::shaders::op
