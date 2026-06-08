/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/src/operators/impl/conv/dilated/dilatedConv.hpp"

namespace mlss::conv::dilated::hip
{

    class HipDilatedConv final : public BackendBase<HipDilatedConv, mlss::conv::dilated::DilatedConv>
    {
    private:

        using base = BackendBase<HipDilatedConv, mlss::conv::dilated::DilatedConv>;

    public:

        HipDilatedConv() = default;

        HipDilatedConv(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipDilatedConv() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context);
    };

} // namespace mlss::conv::dilated::hip
