/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op { class OperatorGEMM; }

namespace mlss::gemm::mxn
{

    class GemmMxN : public CaseBase<GemmMxN, mlss::op::OperatorGEMM>
    {
    private:

        using base = CaseBase<GemmMxN, mlss::op::OperatorGEMM>;

    public:

        GemmMxN() = default;

        GemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~GemmMxN() = default;

        static std::string getCaseName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static uint32_t getCapsImpl(const std::vector<Attribute>& attributes,
                                    GfxIpTriple gfxip,
                                    const void* context);
    };

} // namespace mlss::gemm::mxn
