/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::op
{

    class OperatorRmsNorm : public OperatorBase<OperatorRmsNorm>
    {
    private:

        using base = OperatorBase<OperatorRmsNorm>;

    public:

        OperatorRmsNorm() = default;

        OperatorRmsNorm(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        virtual ~OperatorRmsNorm() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);
    };

} // namespace mlss::op
