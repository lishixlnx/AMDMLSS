/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "../../convMxN.hpp"

namespace mlss::conv::mxn::winograd::ultra
{

    class WinogradUltraConvMxN final : public BackendBase<WinogradUltraConvMxN, mlss::conv::mxn::ConvMxN>
    {
    private:

        using base = BackendBase<WinogradUltraConvMxN, mlss::conv::mxn::ConvMxN>;

    public:

        WinogradUltraConvMxN() = default;

        WinogradUltraConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~WinogradUltraConvMxN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context);
    };

} // namespace mlss::conv::mxn::winograd::ultra
