/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <hip/hip_runtime.h>

#include "test_helpers.hpp"

#include <mlss_tester.hpp>

// ---------------------------------------------------------------------------
// GEMM MxN unit tests (HIP and CK backends).
//
// Coverage:
//   1. Pull compiled binaries through the C-API (mlssGetCaps + mlssGetBinaries).
//   2. Validate that both relocatable and non-relocatable variants are
//      returned, with consistent kernel name + grid metadata.
//   3. Validate the host-side reference GEMM (used as ground truth).
//
// GPU dispatch is intentionally NOT exercised here. The HIP and CK MxN
// shaders are compiled for dxcp's PAL/D3D12 driver path, where each buffer
// kernel argument is delivered as an AMD Buffer Resource Descriptor (SRD)
// rather than a raw 64-bit pointer. They use BUFFER instructions internally
// and rely on PAL to assemble those SRDs (NUM_RECORDS, format, swizzle …)
// from the explicit batchStride arguments.
//
// Launching such kernels through the standard HIP/D3D12 runtime path
// (which delivers raw 64-bit pointers and lets the kernel build its own
// SRDs from FLAT-style addresses) yields silent miscomputation, not a
// crash. Until mlss-tester grows a PAL-style dispatch path - or the
// kernels are recompiled to use FLAT addressing exclusively (as MVN2 /
// MHA / GQA do) - we restrict the tests to the C-API surface.
// ---------------------------------------------------------------------------

namespace
{

// ---------------------------------------------------------------------------
// Per-backend problem sizes — chosen to satisfy each backend's constraints
// while keeping host-reference computation cheap.
// ---------------------------------------------------------------------------

// HIP MxN constraints: M,N,K >= 16; N,K even; alpha == 1.0; FP16 only.
constexpr MLSSuint32 kHipM = 512u;
constexpr MLSSuint32 kHipN = 512u;
constexpr MLSSuint32 kHipK = 512u;

// CK MxN constraints: M,N % 128 == 0, K % 32 == 0, K >= 64, GFX12 only,
// FP32 only.
constexpr MLSSuint32 kCkM = 256u;
constexpr MLSSuint32 kCkN = 256u;
constexpr MLSSuint32 kCkK = 128u;

constexpr MLSSuint32  kBatch       = 1u;
constexpr MLSSfloat32 kAlpha       = 1.0f;
constexpr MLSSfloat32 kBeta        = 0.0f;
constexpr MLSSenum    kActivation  = MLSS_ACTIVATION_IDENTITY;

// Two binaries (relocatable + non-relocatable) per backend.
constexpr std::size_t kExpectedBinaries = 2u;

// Recognized kernel symbol prefixes (mangled C++ names emitted by the
// HIP / Composable-Kernel compilers respectively).
inline constexpr std::string_view kHipGemmKernelPrefix = "_Z9gemm_wmma";
inline constexpr std::string_view kCkGemmKernelPrefix  = "_ZN2ck";

// ---------------------------------------------------------------------------
// MLSS C-API: obtain compiled GEMM binaries for the chosen backend.
// ---------------------------------------------------------------------------

struct MlssBinaries
{
    MLSSbinary* data    = nullptr;
    MLSSsize    count   = 0;
    MLSScontext context = 0;
};

MlssBinaries getGemmBinaries(MLSSstring asic,
                             MLSSuint32 m, MLSSuint32 n, MLSSuint32 k,
                             MLSSenum dataType, MLSSenum precision)
{
    MlssBinaries out{};
    MLSSstring   opName     = const_cast<MLSSstring>(MLSS_GEMM);
    MLSSuint32   batch      = kBatch;
    MLSSfloat32  alpha      = kAlpha;
    MLSSfloat32  beta       = kBeta;
    MLSSbool     hasC       = 0;
    MLSSbool     transA     = 0;
    MLSSbool     transB     = 0;
    MLSSenum     activation = kActivation;

    CHECK_STATUS(mlssCreateContext(&out.context, asic, opName));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_M,          &m));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_N,          &n));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_K,          &k));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_BATCH,      &batch));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_ALPHA,      &alpha));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_BETA,       &beta));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_HASC,       &hasC));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_TRANSA,     &transA));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_TRANSB,     &transB));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_DATATYPE,   &dataType));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_PRECISION,  &precision));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GEMM_ACTIVATION, &activation));

    MLSSstatus* pStatuses = nullptr;
    MLSSsize    nStatuses = 0;
    REQUIRE(mlssGetCaps(out.context, &pStatuses, &nStatuses) == MLSS_SUCCESS);

    CHECK_STATUS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;
}

// ---------------------------------------------------------------------------
// Host-side reference: row-major C = A * B with NN layout (FP32).
// ---------------------------------------------------------------------------

std::vector<float> referenceGemmNN(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   std::uint32_t m, std::uint32_t n, std::uint32_t k)
{
    std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
    th_blas::gemm<float>(
        /*transA=*/false, /*transB=*/false,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        /*alpha=*/1.0f,
        a.data(), static_cast<int>(k),    // A is M x K, lda = K
        b.data(), static_cast<int>(n),    // B is K x N, ldb = N
        /*beta=*/0.0f,
        c.data(), static_cast<int>(n));   // C is M x N, ldc = N
    return c;
}

bool startsWith(std::string_view v, std::string_view prefix)
{
    return v.size() >= prefix.size()
        && v.compare(0, prefix.size(), prefix) == 0;
}

// ---------------------------------------------------------------------------
// Shared test data — built lazily on first access, reused across TEST_CASEs.
// ---------------------------------------------------------------------------

struct GemmTestData
{
    MlssBinaries        hipBins{};
    MlssBinaries        ckBins{};

    std::vector<float>  hostA_hip;   // FP16-quantized FP32 mirror
    std::vector<float>  hostB_hip;
    std::vector<float>  hostRefHip;

    std::vector<float>  hostA_ck;
    std::vector<float>  hostB_ck;
    std::vector<float>  hostRefCk;

    bool gpuAvailable = false;
};

GemmTestData& getTestData()
{
    static GemmTestData data = []
    {
        GemmTestData d;

        hipDeviceProp_t props{};
        d.gpuAvailable = (hipGetDeviceProperties(&props, 0) == hipSuccess);
        if (!d.gpuAvailable) return d;

        std::cout << "Detected GPU: " << props.gcnArchName << '\n';

        MLSSstring asic = const_cast<MLSSstring>(MLSS_GFXAUTOFIND);
        mlssSetVerboseLevel(0);

        d.hipBins = getGemmBinaries(asic, kHipM, kHipN, kHipK,
                                    MLSS_FLOAT16, MLSS_PRECISION_FLOAT16);
        d.ckBins  = getGemmBinaries(asic, kCkM, kCkN, kCkK,
                                    MLSS_FLOAT32, MLSS_PRECISION_FLOAT32);

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        const std::size_t hipANum = static_cast<std::size_t>(kHipM) * kHipK;
        const std::size_t hipBNum = static_cast<std::size_t>(kHipK) * kHipN;
        d.hostA_hip.resize(hipANum);
        d.hostB_hip.resize(hipBNum);
        for (auto& v : d.hostA_hip) v = halfToFloat(floatToHalf(dist(rng)));
        for (auto& v : d.hostB_hip) v = halfToFloat(floatToHalf(dist(rng)));
        d.hostRefHip = referenceGemmNN(d.hostA_hip, d.hostB_hip,
                                       kHipM, kHipN, kHipK);

        const std::size_t ckANum = static_cast<std::size_t>(kCkM) * kCkK;
        const std::size_t ckBNum = static_cast<std::size_t>(kCkK) * kCkN;
        d.hostA_ck.resize(ckANum);
        d.hostB_ck.resize(ckBNum);
        for (auto& v : d.hostA_ck) v = dist(rng);
        for (auto& v : d.hostB_ck) v = dist(rng);
        d.hostRefCk = referenceGemmNN(d.hostA_ck, d.hostB_ck,
                                      kCkM, kCkN, kCkK);

        return d;
    }();
    return data;
}

} // namespace

// ---------------------------------------------------------------------------
// C-API integration: each backend produces two binaries (one relocatable +
// one non-relocatable) carrying the expected mangled kernel name and a
// non-zero dispatch grid.
// ---------------------------------------------------------------------------

TEST_CASE("GEMM HIP MxN: getCaps + getBinaries succeed", "[gemm][hip][api]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(td.hipBins.context != 0);
    REQUIRE(td.hipBins.data    != nullptr);
    REQUIRE(td.hipBins.count    > 0);
}

TEST_CASE("GEMM CK MxN: getCaps + getBinaries succeed", "[gemm][ck][api]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(td.ckBins.context != 0);
    REQUIRE(td.ckBins.data    != nullptr);
    REQUIRE(td.ckBins.count    > 0);
}

TEST_CASE("GEMM HIP MxN: produces reloc + non-reloc binaries with grid metadata",
          "[gemm][hip][api][binaries]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(static_cast<std::size_t>(td.hipBins.count) == kExpectedBinaries);

    std::size_t relocCount    = 0;
    std::size_t nonRelocCount = 0;
    std::size_t gemmKernels   = 0;

    for (MLSSsize i = 0; i < td.hipBins.count; ++i)
    {
        const MLSSbinary& bin = td.hipBins.data[i];
        REQUIRE(bin.m_pKernelName != nullptr);
        REQUIRE(bin.m_binaries    != nullptr);
        REQUIRE(bin.m_binarySize   > 0);
        REQUIRE(bin.m_grid.m_x     > 0u);
        REQUIRE(bin.m_blocks.m_x   > 0u);

        if (bin.m_isRelocatable) ++relocCount;
        else                     ++nonRelocCount;

        if (startsWith(bin.m_pKernelName, kHipGemmKernelPrefix)) ++gemmKernels;
    }

    CHECK(relocCount    == 1u);
    CHECK(nonRelocCount == 1u);
    CHECK(gemmKernels   == 2u);
}

TEST_CASE("GEMM CK MxN: produces reloc + non-reloc binaries with grid metadata",
          "[gemm][ck][api][binaries]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(static_cast<std::size_t>(td.ckBins.count) == kExpectedBinaries);

    std::size_t relocCount    = 0;
    std::size_t nonRelocCount = 0;
    std::size_t ckKernels     = 0;

    for (MLSSsize i = 0; i < td.ckBins.count; ++i)
    {
        const MLSSbinary& bin = td.ckBins.data[i];
        REQUIRE(bin.m_pKernelName != nullptr);
        REQUIRE(bin.m_binaries    != nullptr);
        REQUIRE(bin.m_binarySize   > 0);
        REQUIRE(bin.m_grid.m_x     > 0u);
        REQUIRE(bin.m_blocks.m_x   > 0u);

        if (bin.m_isRelocatable) ++relocCount;
        else                     ++nonRelocCount;

        if (startsWith(bin.m_pKernelName, kCkGemmKernelPrefix)) ++ckKernels;
    }

    CHECK(relocCount    == 1u);
    CHECK(nonRelocCount == 1u);
    CHECK(ckKernels     == 2u);
}

TEST_CASE("GEMM: each binary advertises MLSS_GEMM as its operator", "[gemm][api]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");

    auto checkOpName = [](const MlssBinaries& bins)
    {
        for (MLSSsize i = 0; i < bins.count; ++i)
        {
            const MLSSbinary& bin = bins.data[i];
            REQUIRE(bin.m_pOperatorName != nullptr);
            REQUIRE(bin.m_ASIC          != nullptr);
            CHECK(std::string_view(bin.m_pOperatorName) == "MLSS_GEMM");
        }
    };
    checkOpName(td.hipBins);
    checkOpName(td.ckBins);
}

TEST_CASE("GEMM HIP MxN: relocatable and non-relocatable variants share kernel symbol",
          "[gemm][hip][api][binaries]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");

    const auto* reloc    = findBinaryByKernelPrefix(td.hipBins.data, td.hipBins.count,
                                                    kHipGemmKernelPrefix, /*relocatable=*/true);
    const auto* nonReloc = findBinaryByKernelPrefix(td.hipBins.data, td.hipBins.count,
                                                    kHipGemmKernelPrefix, /*relocatable=*/false);
    REQUIRE(reloc    != nullptr);
    REQUIRE(nonReloc != nullptr);

    CHECK(std::string_view(reloc->m_pKernelName)
       == std::string_view(nonReloc->m_pKernelName));

    CHECK(reloc->m_grid.m_x   == nonReloc->m_grid.m_x);
    CHECK(reloc->m_grid.m_y   == nonReloc->m_grid.m_y);
    CHECK(reloc->m_grid.m_z   == nonReloc->m_grid.m_z);
    CHECK(reloc->m_blocks.m_x == nonReloc->m_blocks.m_x);
    CHECK(reloc->m_blocks.m_y == nonReloc->m_blocks.m_y);
    CHECK(reloc->m_blocks.m_z == nonReloc->m_blocks.m_z);
}

TEST_CASE("GEMM CK MxN: relocatable and non-relocatable variants share kernel symbol",
          "[gemm][ck][api][binaries]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");

    const auto* reloc    = findBinaryByKernelPrefix(td.ckBins.data, td.ckBins.count,
                                                    kCkGemmKernelPrefix, /*relocatable=*/true);
    const auto* nonReloc = findBinaryByKernelPrefix(td.ckBins.data, td.ckBins.count,
                                                    kCkGemmKernelPrefix, /*relocatable=*/false);
    REQUIRE(reloc    != nullptr);
    REQUIRE(nonReloc != nullptr);

    CHECK(std::string_view(reloc->m_pKernelName)
       == std::string_view(nonReloc->m_pKernelName));

    CHECK(reloc->m_grid.m_x   == nonReloc->m_grid.m_x);
    CHECK(reloc->m_grid.m_y   == nonReloc->m_grid.m_y);
    CHECK(reloc->m_grid.m_z   == nonReloc->m_grid.m_z);
    CHECK(reloc->m_blocks.m_x == nonReloc->m_blocks.m_x);
    CHECK(reloc->m_blocks.m_y == nonReloc->m_blocks.m_y);
    CHECK(reloc->m_blocks.m_z == nonReloc->m_blocks.m_z);
}

// ---------------------------------------------------------------------------
// Host-side ground-truth sanity check.
// ---------------------------------------------------------------------------

TEST_CASE("GEMM HIP MxN: host reference is well-formed", "[gemm][hip][host]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");

    REQUIRE(td.hostRefHip.size() == static_cast<std::size_t>(kHipM) * kHipN);
    for (float v : td.hostRefHip) REQUIRE(std::isfinite(v));
}

TEST_CASE("GEMM CK MxN: host reference is well-formed", "[gemm][ck][host]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");

    REQUIRE(td.hostRefCk.size() == static_cast<std::size_t>(kCkM) * kCkN);
    for (float v : td.hostRefCk) REQUIRE(std::isfinite(v));
}
