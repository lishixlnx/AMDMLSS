/* Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::op
{

    class OperatorGEMM : public OperatorBase<OperatorGEMM>
    {
    private:

        using base = OperatorBase<OperatorGEMM>;

        friend class OperatorBase<OperatorGEMM>;

    public:

        OperatorGEMM() = default;

        OperatorGEMM(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        virtual ~OperatorGEMM() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);
    };

} // namespace mlss::op
