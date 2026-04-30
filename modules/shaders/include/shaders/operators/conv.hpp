/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::op
{
    class OperatorConv : public OperatorBase<OperatorConv>
    {
    private:

        using base = OperatorBase<OperatorConv>;

    public:

        OperatorConv() = default;

        OperatorConv(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        virtual ~OperatorConv() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);
    };

} // namespace mlss::op
