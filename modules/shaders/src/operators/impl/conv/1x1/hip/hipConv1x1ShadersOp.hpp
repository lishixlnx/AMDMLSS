/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "../conv1x1.hpp"

namespace mlss::conv::one_by_one::hip::wmma
{

    class HipConv1x1 final : public BackendBase<HipConv1x1, mlss::conv::one_by_one::Conv1x1>
    {
    private:

        using base = BackendBase<HipConv1x1, mlss::conv::one_by_one::Conv1x1>;

    public:

    HipConv1x1() = default;

    HipConv1x1(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipConv1x1() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context);
    };

} // namespace mlss::conv::one_by_one::hip::wmma
