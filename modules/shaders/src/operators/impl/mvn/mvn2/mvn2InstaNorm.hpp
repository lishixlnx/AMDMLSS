/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorMVN; }

namespace mlss::mvn::mvn2
{

    class MVN2InstaNorm : public CaseBase<MVN2InstaNorm, mlss::op::OperatorMVN>
    {
    private:

        using base = CaseBase<MVN2InstaNorm, mlss::op::OperatorMVN>;

    public:

        MVN2InstaNorm() = default;

        MVN2InstaNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~MVN2InstaNorm() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context);
    };

} // namespace mlss::mvn::mvn2
