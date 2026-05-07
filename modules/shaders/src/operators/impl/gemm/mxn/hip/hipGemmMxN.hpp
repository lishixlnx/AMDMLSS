/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::gemm::mxn { class GemmMxN; }

namespace mlss::gemm::mxn::hip
{

    class HipGemmMxN final : public BackendBase<HipGemmMxN, mlss::gemm::mxn::GemmMxN>
    {
    private:

        using base = BackendBase<HipGemmMxN, mlss::gemm::mxn::GemmMxN>;

    public:

        HipGemmMxN() = default;

        HipGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipGemmMxN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes,
                                    const GfxIpTriple& gfxArch,
                                    const void* context);
    };

} // namespace mlss::gemm::mxn::hip
