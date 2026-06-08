/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorGemmGemm; }

namespace mlss::gemm_gemm::mxn
{

    class GemmGemmMxN : public CaseBase<GemmGemmMxN, mlss::op::OperatorGemmGemm>
    {
    private:

        using base = CaseBase<GemmGemmMxN, mlss::op::OperatorGemmGemm>;

    public:

        GemmGemmMxN() = default;

        GemmGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~GemmGemmMxN() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context);
    };

} // namespace mlss::gemm_gemm::mxn
