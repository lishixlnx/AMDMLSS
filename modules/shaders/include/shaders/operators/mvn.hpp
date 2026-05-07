/* Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::op
{

    class OperatorMVN : public OperatorBase<OperatorMVN>
    {
    private:

        using base = OperatorBase<OperatorMVN>;

    public:

        OperatorMVN() = default;

        OperatorMVN(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        virtual ~OperatorMVN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);
    };

} // namespace mlss::op
