/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::gemm_gemm::mxn { class GemmGemmMxN; }

namespace mlss::gemm_gemm::mxn::hip
{

    class HipGemmGemmMxN final : public BackendBase<HipGemmGemmMxN, mlss::gemm_gemm::mxn::GemmGemmMxN>
    {
    private:

        using base = BackendBase<HipGemmGemmMxN, mlss::gemm_gemm::mxn::GemmGemmMxN>;

    public:

        HipGemmGemmMxN() = default;

        HipGemmGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipGemmGemmMxN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::gemm_gemm::mxn::hip
