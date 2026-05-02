/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::gemm::mxn { class GemmMxN; }

namespace mlss::gemm::mxn::ck
{

    class CKGemmMxN final : public BackendBase<CKGemmMxN, mlss::gemm::mxn::GemmMxN>
    {
    private:

        using base = BackendBase<CKGemmMxN, mlss::gemm::mxn::GemmMxN>;

    public:

        CKGemmMxN() = default;

        CKGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~CKGemmMxN() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::gemm::mxn::ck
