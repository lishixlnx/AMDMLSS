/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "../conv1x1.hpp"

namespace mlss::conv::one_by_one::misa
{

    class MisaConv1x1 final : public BackendBase<MisaConv1x1, mlss::conv::one_by_one::Conv1x1>
    {
    private:

        using base = BackendBase<MisaConv1x1, mlss::conv::one_by_one::Conv1x1>;

    public:

        MisaConv1x1() = default;

        MisaConv1x1(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~MisaConv1x1() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context);
    };

} // namespace mlss::conv::one_by_one::misa
