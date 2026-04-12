/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorConv; }

namespace mlss::conv::mxn
{

    class ConvMxN : public CaseBase<ConvMxN, mlss::op::OperatorConv>
    {
    private:

        using base = CaseBase<ConvMxN, mlss::op::OperatorConv>;

    public:

        ConvMxN() = default;

        ConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~ConvMxN() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context);
    };

} // namespace mlss::conv::mxn
