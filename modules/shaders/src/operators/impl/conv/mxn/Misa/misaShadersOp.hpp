/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/src/operators/impl/conv/mxn/convMxN.hpp"

namespace mlss::conv::mxn::misa
{

    class MisaConvMxN final : public BackendBase<MisaConvMxN, mlss::conv::mxn::ConvMxN>
    {
    private:

        using base = BackendBase<MisaConvMxN, mlss::conv::mxn::ConvMxN>;

    public:

        MisaConvMxN() = default;

        MisaConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~MisaConvMxN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch, const void* context);
    };

} // namespace mlss::conv::mxn::misa
