/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::sigmoid_mul { class SigmoidMul; }

namespace mlss::sigmoid_mul::hip
{

    class HipSigmoidMul final : public BackendBase<HipSigmoidMul, mlss::sigmoid_mul::SigmoidMul>
    {
    private:

        using base = BackendBase<HipSigmoidMul, mlss::sigmoid_mul::SigmoidMul>;

    public:

        HipSigmoidMul() = default;

        HipSigmoidMul(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipSigmoidMul() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::sigmoid_mul::hip
