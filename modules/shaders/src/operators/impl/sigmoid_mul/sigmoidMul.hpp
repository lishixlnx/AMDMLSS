/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorSigmoidMul; }

namespace mlss::sigmoid_mul
{

    class SigmoidMul : public CaseBase<SigmoidMul, mlss::op::OperatorSigmoidMul>
    {
    private:

        using base = CaseBase<SigmoidMul, mlss::op::OperatorSigmoidMul>;

    public:

        SigmoidMul() = default;

        SigmoidMul(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~SigmoidMul() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context);
    };

} // namespace mlss::sigmoid_mul
