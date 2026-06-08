/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"

namespace mlss::op
{

    class OperatorGemmGemm : public OperatorBase<OperatorGemmGemm>
    {
    private:

        using base = OperatorBase<OperatorGemmGemm>;

    public:

        OperatorGemmGemm() = default;

        OperatorGemmGemm(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        virtual ~OperatorGemmGemm() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);
    };

} // namespace mlss::op
