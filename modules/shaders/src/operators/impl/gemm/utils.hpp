/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"
#include "../opUtils.hpp"

namespace mlss::gemm::utils
{

    // Defines the full set of parameters to a generic matrix multiply (GEMM).
    // Mirrors the dxcp 'GenericGemmParams' so a single parsed structure can
    // be forwarded between the operator, its cases and its backends without
    // re-parsing the attribute vector.
    struct GenericGemmParams
    {
        std::uint32_t m;
        std::uint32_t n;
        std::uint32_t k;
        std::uint32_t batch;
        float         alpha;
        float         beta;
        bool          hasC;
        bool          transA;
        bool          transB;

        DataTypeFlags           dataType;
        PrecisionFlags          precision;
        ActivationFunctionFlags activation;
    };

    GenericGemmParams buildGemmParams(const std::vector<Attribute>& attributes);

    // Hyperboloid heuristic. Returns true when the (m, n, k) point lies inside
    // the per-architecture surface that approximates 'we are faster than MS shaders'.
    bool chooseTuning(const GfxIpTriple& gfxip, const GenericGemmParams& params);

} // namespace mlss::gemm::utils
