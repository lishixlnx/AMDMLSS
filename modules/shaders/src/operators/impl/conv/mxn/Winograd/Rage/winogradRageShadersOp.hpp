/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/src/operators/impl/conv/mxn/convMxN.hpp"

namespace mlss::conv::mxn::winograd::rage
{

    class WinogradRageConvMxN final : public BackendBase<WinogradRageConvMxN, mlss::conv::mxn::ConvMxN>
    {
    private:

        using base_t = BackendBase<WinogradRageConvMxN, mlss::conv::mxn::ConvMxN>;

    public:

        WinogradRageConvMxN() = default;

        WinogradRageConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~WinogradRageConvMxN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context);
    };

} // namespace mlss::conv::mxn::winograd::rage
