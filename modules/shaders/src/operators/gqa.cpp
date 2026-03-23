/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/gqa.hpp"

namespace mlss::shaders::op
{
      
        //=====================================================================================================================
        // OperatorGQA implementation
        //=====================================================================================================================

        OperatorGQA::OperatorGQA(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
            : base(attributes, gfxip)
        {
        }

        std::string OperatorGQA::getOperatorName()
        {
            return "AMDMLSS::OperatorMHA1_t";
        }

        std::expected<OperatorGQA::blob, std::error_code> OperatorGQA::getBlob() const
        {
            return std::unexpected(std::make_error_code(std::errc::operation_not_supported));
        }

        bool OperatorGQA::getCapsImpl(const std::vector<mlss::Attribute>& attributes)
        {
            return false;
        }

} // namespace mlss::shaders::op
