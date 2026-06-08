/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorRmsNorm; }

namespace mlss::rmsnorm
{

    class RmsNorm : public CaseBase<RmsNorm, mlss::op::OperatorRmsNorm>
    {
    private:

        using base = CaseBase<RmsNorm, mlss::op::OperatorRmsNorm>;

    public:

        RmsNorm() = default;

        RmsNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~RmsNorm() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context);
    };

} // namespace mlss::rmsnorm
