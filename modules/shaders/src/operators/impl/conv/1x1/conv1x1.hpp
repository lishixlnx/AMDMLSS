/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorConv; }

namespace mlss::conv::one_by_one
{

    class Conv1x1 : public CaseBase<Conv1x1, mlss::op::OperatorConv>
    {
    private:

        using base = CaseBase<Conv1x1, mlss::op::OperatorConv>;

    public:

        Conv1x1() = default;

        Conv1x1(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~Conv1x1() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context);
    };

} // namespace mlss::conv::one_by_one
