/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <hip/hip_runtime.h>

#include "test_helpers.hpp"

#include <mlss_tester.hpp>

namespace
{

// ---------------------------------------------------------------------------
// MVN2 (InstaNorm) constraints — see modules/shaders/.../mvn2/hip:
//   - N must be 1
//   - dataType is FLOAT16
//   - crossChannel must be false (per-channel normalization)
//   - (H * W) is a multiple of 256
//   - hasScale and hasBias both true
//   - activation is IDENTITY
//
// We pick (C=8, H=W=32) → HW=1024 which is divisible by both 256 (kernel#1
// tile) and 128 (kernel#3 tile) so all three kernels dispatch a non-trivial
// number of blocks per channel.
// ---------------------------------------------------------------------------

constexpr MLSSuint32  kBatchSize    = 1u;
constexpr MLSSuint32  kChannels     = 8u;
constexpr MLSSuint32  kHeight       = 32u;
constexpr MLSSuint32  kWidth        = 32u;
constexpr MLSSfloat32 kEpsilon      = 1e-5f;
constexpr float       kTolerance    = 5e-2f;
constexpr std::size_t kSpatial      = static_cast<std::size_t>(kHeight) * kWidth;
constexpr std::size_t kElementCount =
    static_cast<std::size_t>(kBatchSize) * kChannels * kSpatial;
constexpr std::size_t kExpectedKernels  = 3u;
constexpr std::size_t kExpectedBinaries = 2u * kExpectedKernels;

// Dispatch tile sizes — must mirror modules/shaders/.../mvn2/hip/shadersUtils.cpp
constexpr std::uint32_t kKernel1Tile  = 256u;
constexpr std::uint32_t kKernel3Tile  = 128u;

// The MVN2 kernels carry C++-mangled symbols inside the ELF
// (e.g. "_Z28norm_nchw_split_kernel_part1PPKDF16_PPfjj"). The prefix below
// uniquely identifies kernel #1 / #2 / #3 across all GPU targets.
inline constexpr std::string_view kMvn2KernelPrefix1 = "_Z28norm_nchw_split_kernel_part1";
inline constexpr std::string_view kMvn2KernelPrefix2 = "_Z28norm_nchw_split_kernel_part2";
inline constexpr std::string_view kMvn2KernelPrefix3 = "_Z28norm_nchw_split_kernel_part3";

// Broadcast type 'Nhw' (per-channel scale/bias broadcast over N, H, W).
constexpr std::uint32_t kBrdcstTypeNhw = 2u;

// ---------------------------------------------------------------------------
// MLSS C-API: obtain compiled MVN2 (InstaNorm) binaries.
// ---------------------------------------------------------------------------

struct MlssBinaries
{
    MLSSbinary* data    = nullptr;
    MLSSsize    count   = 0;
    MLSScontext context = 0;
};

MlssBinaries getMvn2Binaries(MLSSstring asic)
{
    MlssBinaries out{};
    MLSSstring   opName       = const_cast<MLSSstring>(MLSS_MVN);
    MLSSuint32   batch        = kBatchSize;
    MLSSuint32   channels     = kChannels;
    MLSSuint32   height       = kHeight;
    MLSSuint32   width        = kWidth;
    MLSSfloat32  epsilon      = kEpsilon;
    MLSSbool     crossChannel = 0;
    MLSSuint32   sbDims[4]    = { 1u, channels, 1u, 1u };
    MLSSbool     hasScale     = 1;
    MLSSbool     hasBias      = 1;
    MLSSenum     dataType     = MLSS_FLOAT16;
    MLSSenum     activation   = MLSS_ACTIVATION_IDENTITY;

    CHECK_STATUS(mlssCreateContext(&out.context, asic, opName));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_N,             &batch));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_C,             &channels));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_H,             &height));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_W,             &width));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_EPSILON,       &epsilon));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_CROSSCHANNEL,  &crossChannel));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_SBDIMS,        sbDims));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_HASSCALE,      &hasScale));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_HASBIAS,       &hasBias));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_DATATYPE,      &dataType));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MVN_ACTIVATION,    &activation));

    MLSSstatus* pStatuses = nullptr;
    MLSSsize    nStatuses = 0;
    REQUIRE(mlssGetCaps(out.context, &pStatuses, &nStatuses) == MLSS_SUCCESS);

    CHECK_STATUS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;
}

// ---------------------------------------------------------------------------
// Host-side reference InstaNorm:
//   For each (batch, channel) pair, compute mean/variance over the H*W
//   spatial extent, then normalize and apply per-channel scale + bias.
// ---------------------------------------------------------------------------

std::vector<float> referenceInstaNorm(const std::vector<float>& input,
                                      const std::vector<float>& scale,
                                      const std::vector<float>& bias,
                                      float epsilon)
{
    std::vector<float> output(input.size(), 0.0f);

    for (std::uint32_t b = 0; b < kBatchSize; ++b)
    {
        for (std::uint32_t ch = 0; ch < kChannels; ++ch)
        {
            const std::size_t base = (static_cast<std::size_t>(b) * kChannels + ch) * kSpatial;

            float mean = 0.0f;
            for (std::size_t s = 0; s < kSpatial; ++s)
            {
                mean += input[base + s];
            }
            mean /= static_cast<float>(kSpatial);

            float variance = 0.0f;
            for (std::size_t s = 0; s < kSpatial; ++s)
            {
                const float diff = input[base + s] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(kSpatial);

            const float invStd = 1.0f / std::sqrt(variance + epsilon);
            const float gamma  = scale[ch];
            const float beta   = bias[ch];

            for (std::size_t s = 0; s < kSpatial; ++s)
            {
                output[base + s] = (input[base + s] - mean) * invStd * gamma + beta;
            }
        }
    }

    return output;
}

bool startsWith(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size()
        && value.compare(0, prefix.size(), prefix) == 0;
}

// Shared test data — built once across all TEST_CASEs.
struct Mvn2TestData
{
    MlssBinaries           bins{};
    std::vector<float>     hostInput;
    std::vector<float>     hostScale;
    std::vector<float>     hostBias;
    std::vector<float>     hostRef;
    std::vector<float16_t> hostInput16;
    std::vector<float16_t> hostScale16;
    std::vector<float16_t> hostBias16;
    bool                   gpuAvailable = false;
};

Mvn2TestData& getTestData()
{
    static Mvn2TestData data = []
    {
        Mvn2TestData d;

        hipDeviceProp_t props{};
        d.gpuAvailable = (hipGetDeviceProperties(&props, 0) == hipSuccess);
        if (!d.gpuAvailable) return d;

        std::cout << "Detected GPU: " << props.gcnArchName << '\n';

        MLSSstring asic = const_cast<MLSSstring>(MLSS_GFXAUTOFIND);
        mlssSetVerboseLevel(0);

        d.bins = getMvn2Binaries(asic);

        d.hostInput.resize(kElementCount);
        d.hostScale.assign(kChannels, 1.0f);
        d.hostBias.assign(kChannels, 0.0f);

        // Per-channel deterministic ramp so each channel sees a different
        // mean/variance — exercises the full normalize + scale + bias path.
        for (std::uint32_t b = 0; b < kBatchSize; ++b)
        {
            for (std::uint32_t ch = 0; ch < kChannels; ++ch)
            {
                const std::size_t base = (static_cast<std::size_t>(b) * kChannels + ch) * kSpatial;
                for (std::size_t s = 0; s < kSpatial; ++s)
                {
                    d.hostInput[base + s] = static_cast<float>(ch) * 0.5f
                                          + static_cast<float>(s) * 0.01f;
                }
            }
        }

        // Non-trivial scale/bias to make sure the affine step is checked.
        for (std::uint32_t ch = 0; ch < kChannels; ++ch)
        {
            d.hostScale[ch] = 1.0f + 0.1f * static_cast<float>(ch);
            d.hostBias[ch]  = -0.05f * static_cast<float>(ch);
        }

        d.hostRef = referenceInstaNorm(d.hostInput, d.hostScale, d.hostBias, kEpsilon);

        // FP16 mirrors of the host buffers used to feed each backend.
        d.hostInput16.resize(d.hostInput.size());
        d.hostScale16.resize(d.hostScale.size());
        d.hostBias16.resize(d.hostBias.size());
        for (std::size_t i = 0; i < d.hostInput.size(); ++i)
            d.hostInput16[i] = floatToHalf(d.hostInput[i]);
        for (std::size_t i = 0; i < d.hostScale.size(); ++i)
            d.hostScale16[i] = floatToHalf(d.hostScale[i]);
        for (std::size_t i = 0; i < d.hostBias.size(); ++i)
            d.hostBias16[i]  = floatToHalf(d.hostBias[i]);

        return d;
    }();
    return data;
}

// ---------------------------------------------------------------------------
// Generic GPU pipeline runner.
//
// Resolves three kernels (instaNormSplitPart{1,2,3}) from the binary set,
// allocates input / scale / bias / temp / output device buffers, builds the
// kernel-arg lists by hand (the MVN2 ELFs do not carry self-describing arg
// metadata) and dispatches the three-stage InstaNorm pipeline.  Returns the
// downloaded fp16 output as a vector of floats.
//
// Layout of the temporary buffer (uint32 element count, total tempBytes):
//   [0 .. kernel2OutElems)                     mean & std (kernel#2 output)
//   [kernel2OutElems .. kernel2OutElems +
//                       kernel1OutElems)       partial sums (kernel#1 output,
//                                              kernel#2 input)
// ---------------------------------------------------------------------------

template <typename Module, typename Memory, typename Shader>
std::vector<float> runMvn2Pipeline(MLSSbinary*      binaries,
                                   MLSSsize         count,
                                   bool             relocatable,
                                   const std::vector<float16_t>& hostInput,
                                   const std::vector<float16_t>& hostScale,
                                   const std::vector<float16_t>& hostBias)
{
    const auto* binPart1 = findBinaryByKernelPrefix(binaries, count, kMvn2KernelPrefix1, relocatable);
    const auto* binPart2 = findBinaryByKernelPrefix(binaries, count, kMvn2KernelPrefix2, relocatable);
    const auto* binPart3 = findBinaryByKernelPrefix(binaries, count, kMvn2KernelPrefix3, relocatable);
    REQUIRE(binPart1 != nullptr);
    REQUIRE(binPart2 != nullptr);
    REQUIRE(binPart3 != nullptr);

    auto desc1 = buildShaderDescriptor(*binPart1);
    auto desc2 = buildShaderDescriptor(*binPart2);
    auto desc3 = buildShaderDescriptor(*binPart3);

    // --- module construction (per kernel — backends require a dedicated
    //     program/library per ELF) -------------------------------------------
    [[maybe_unused]] ClContext backendCtx{};

    auto makeModule = [&](const ShaderDescriptor& desc) -> Module
    {
        if constexpr (std::is_same_v<Module, ClManagedModule>)
            return Module(backendCtx, desc);
        else
            return Module(desc);
    };

    Module m1 = makeModule(desc1);
    Module m2 = makeModule(desc2);
    Module m3 = makeModule(desc3);
    REQUIRE(m1.isLoaded());
    REQUIRE(m2.isLoaded());
    REQUIRE(m3.isLoaded());

    // The MVN2 ELF kernels are mangled (e.g. "_Z28norm_nchw_split_kernel_part1...")
    // and don't match the descriptive m_pKernelName the C-API exposes. Each
    // ELF holds exactly one global function, so enumerate the module's
    // shader names and pick the first one.
    auto resolveShader = [&](Module& module, const MLSSbinary& bin) -> Shader
    {
        const auto names = module.getShadersNames();
        const std::string lookupName = !names.empty() ? names.front()
                                     : (bin.m_pKernelName ? std::string(bin.m_pKernelName) : std::string{});

        if constexpr (std::is_same_v<Shader, D3D12Shader>)
        {
            const BaseShaderTag* tag = module.getShader(lookupName);
            if (tag == nullptr && bin.m_pKernelName != nullptr)
                tag = module.getShader(bin.m_pKernelName);
            REQUIRE(tag != nullptr);
            return *static_cast<const D3D12Shader*>(tag);
        }
        else
        {
            Shader sh = module.getShader(lookupName);
            if (!sh.isValid() && bin.m_pKernelName != nullptr)
                sh = module.getShader(bin.m_pKernelName);
            return sh;
        }
    };

    Shader k1 = resolveShader(m1, *binPart1);
    Shader k2 = resolveShader(m2, *binPart2);
    Shader k3 = resolveShader(m3, *binPart3);
    REQUIRE(k1.isValid());
    REQUIRE(k2.isValid());
    REQUIRE(k3.isValid());

    // --- memory allocation -------------------------------------------------
    constexpr std::size_t fp16Bytes = sizeof(float16_t);

    auto allocMem = [&](std::size_t bytes) -> Memory
    {
        if constexpr (std::is_same_v<Memory, D3D12DeviceMemory>)
            return Memory(m1.getDevice()->handle().Get(), bytes);
        else if constexpr (std::is_same_v<Memory, ClDeviceMemory>)
            return Memory(backendCtx, bytes);
        else
            return Memory(bytes);
    };

    // Per dxcp ddiMetaCmdMvn2.cpp:
    //   kernel1OutElems = 2 * (HW / kernel1TileSize) * C   (uint32 elements)
    //   kernel2OutElems = 2 * C                            (uint32 elements)
    //   tempBytes       = (kernel1OutElems + kernel2OutElems) * sizeof(uint32)
    const std::uint32_t hw              = kHeight * kWidth;
    const std::uint32_t kernel1OutElems = 2u * (hw / kKernel1Tile) * kChannels;
    const std::uint32_t kernel2OutElems = 2u * kChannels;
    const std::size_t   tempBytes       =
        static_cast<std::size_t>(kernel1OutElems + kernel2OutElems) * sizeof(std::uint32_t);

    Memory devInput = allocMem(fp16Bytes * kElementCount);
    Memory devScale = allocMem(fp16Bytes * kChannels);
    Memory devBias  = allocMem(fp16Bytes * kChannels);
    Memory devTemp  = allocMem(tempBytes);
    Memory devOut   = allocMem(fp16Bytes * kElementCount);

    devInput.upload(hostInput);
    devScale.upload(hostScale);
    devBias .upload(hostBias);

    devInput.setDoublePointer();
    devScale.setDoublePointer();
    devBias .setDoublePointer();
    devTemp .setDoublePointer();
    devOut  .setDoublePointer();

    // --- dispatch dimensions (must mirror shadersUtils.cpp) ----------------
    const dim3 grid1(hw / kKernel1Tile, kChannels, 1);
    const dim3 grid2(kChannels, 1, 1);
    const dim3 grid3(hw / kKernel3Tile, kChannels, 1);
    const dim3 block1(64, 1, 1);
    const dim3 block2(32, 1, 1);
    const dim3 block3(32, 1, 1);

    // --- kernel#1 args (input → temp[partialOffset:]) ----------------------
    const std::uint32_t hwArg            = hw;
    const std::uint32_t partialOffset    = kernel2OutElems;
    {
        std::vector<KernelArg> args;
        args.push_back(KernelArg(devInput.getDoublePointer())); // in: input tensor (NCHW)
        args.push_back(KernelArg(devTemp .getDoublePointer())); // out: temp buffer base
        args.push_back(KernelArg(partialOffset));               // out offset (in u32 elems)
        args.push_back(KernelArg(hwArg));                       // const: H * W
        REQUIRE(k1.run(args, grid1, block1));
    }

    // --- kernel#2 args (temp[partialOffset:] → temp[0:meanStdSize]) --------
    {
        const std::uint32_t partialSumElemCnt = (hw / kKernel1Tile) * kChannels;
        const std::uint32_t channelsArg       = kChannels;
        const float         epsArg            = kEpsilon;

        std::vector<KernelArg> args;
        args.push_back(KernelArg(devTemp.getDoublePointer())); // in: partial sums
        args.push_back(KernelArg(devTemp.getDoublePointer())); // out: mean & std
        args.push_back(KernelArg(partialOffset));              // in offset (in u32 elems)
        args.push_back(KernelArg(partialSumElemCnt));          // const: # partial-sum entries
        args.push_back(KernelArg(hwArg));                      // const: H * W
        args.push_back(KernelArg(channelsArg));                // const: # channels
        args.push_back(KernelArg(epsArg));                     // const: epsilon
        REQUIRE(k2.run(args, grid2, block2));
    }

    // --- kernel#3 args (input + scale + bias + temp[0:] → output) ----------
    {
        const std::uint32_t brdcstType = kBrdcstTypeNhw;

        std::vector<KernelArg> args;
        args.push_back(KernelArg(devInput.getDoublePointer())); // in: input
        args.push_back(KernelArg(devScale.getDoublePointer())); // in: scale
        args.push_back(KernelArg(devBias .getDoublePointer())); // in: bias
        args.push_back(KernelArg(devTemp .getDoublePointer())); // in: mean & std
        args.push_back(KernelArg(devOut  .getDoublePointer())); // out: result
        args.push_back(KernelArg(hwArg));                       // const: H * W
        args.push_back(KernelArg(brdcstType));                  // const: broadcast type
        REQUIRE(k3.run(args, grid3, block3));
    }

    // --- readback ----------------------------------------------------------
    std::vector<float16_t> outFp16(kElementCount);
    devOut.download(outFp16);
    return halvesToFloats(outFp16);
}

bool resultsMatch(const std::vector<float>& a, const std::vector<float>& b,
                  float tolerance, std::string_view label)
{
    REQUIRE(a.size() == b.size());

    bool        pass         = true;
    std::size_t mismatchCount = 0;
    constexpr std::size_t kMaxPrinted = 10;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > tolerance)
        {
            if (mismatchCount < kMaxPrinted)
            {
                std::cerr << label << ": mismatch at [" << i << "] "
                          << a[i] << " vs " << b[i] << " diff=" << diff << '\n';
            }
            ++mismatchCount;
            pass = false;
        }
    }

    if (mismatchCount > kMaxPrinted)
    {
        std::cerr << label << ": ... and " << (mismatchCount - kMaxPrinted)
                  << " more mismatches\n";
    }
    return pass;
}

} // namespace

// ---------------------------------------------------------------------------
// C-API integration tests: validate that mlssGetCaps + mlssGetBinaries
// return a well-formed binary set with both relocatable and non-relocatable
// flavors of each of the three InstaNorm kernels.
// ---------------------------------------------------------------------------

TEST_CASE("MVN2: getCaps and getBinaries succeed", "[mvn2][api]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(td.bins.context != 0);
    REQUIRE(td.bins.data != nullptr);
    REQUIRE(td.bins.count > 0);
}

TEST_CASE("MVN2: produces three kernels in both reloc and non-reloc flavors",
          "[mvn2][api][binaries]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(static_cast<std::size_t>(td.bins.count) == kExpectedBinaries);

    std::size_t relocCount    = 0;
    std::size_t nonRelocCount = 0;
    std::size_t part1Count    = 0;
    std::size_t part2Count    = 0;
    std::size_t part3Count    = 0;

    for (MLSSsize i = 0; i < td.bins.count; ++i)
    {
        const MLSSbinary& bin = td.bins.data[i];
        REQUIRE(bin.m_pKernelName != nullptr);
        REQUIRE(bin.m_binaries != nullptr);
        REQUIRE(bin.m_binarySize > 0);
        REQUIRE(bin.m_grid.m_x > 0u);
        REQUIRE(bin.m_blocks.m_x > 0u);

        if (bin.m_isRelocatable) ++relocCount;
        else                     ++nonRelocCount;

        const std::string_view name(bin.m_pKernelName);
        if (startsWith(name, kMvn2KernelPrefix1)) ++part1Count;
        if (startsWith(name, kMvn2KernelPrefix2)) ++part2Count;
        if (startsWith(name, kMvn2KernelPrefix3)) ++part3Count;
    }

    CHECK(relocCount    == kExpectedKernels);
    CHECK(nonRelocCount == kExpectedKernels);
    CHECK(part1Count    == 2u);
    CHECK(part2Count    == 2u);
    CHECK(part3Count    == 2u);
}

TEST_CASE("MVN2: each binary advertises the correct operator and ASIC",
          "[mvn2][api]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");

    for (MLSSsize i = 0; i < td.bins.count; ++i)
    {
        const MLSSbinary& bin = td.bins.data[i];
        REQUIRE(bin.m_pOperatorName != nullptr);
        REQUIRE(bin.m_ASIC          != nullptr);
        CHECK(std::string_view(bin.m_pOperatorName) == "MLSS_MVN");
    }
}

// ---------------------------------------------------------------------------
// Host reference sanity check.
// ---------------------------------------------------------------------------

TEST_CASE("MVN2: Host reference is well-formed", "[mvn2][host]")
{
    auto& td = getTestData();
    if (!td.gpuAvailable) SKIP("No compatible GPU detected");
    REQUIRE(td.hostRef.size() == kElementCount);

    for (float v : td.hostRef)
    {
        REQUIRE(std::isfinite(v));
    }

    for (std::uint32_t b = 0; b < kBatchSize; ++b)
    {
        for (std::uint32_t ch = 0; ch < kChannels; ++ch)
        {
            const std::size_t base = (static_cast<std::size_t>(b) * kChannels + ch) * kSpatial;

            float mean = 0.0f;
            for (std::size_t s = 0; s < kSpatial; ++s)
                mean += td.hostRef[base + s];
            mean /= static_cast<float>(kSpatial);

            float variance = 0.0f;
            for (std::size_t s = 0; s < kSpatial; ++s)
            {
                const float diff = td.hostRef[base + s] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(kSpatial);

            const float expectedMean = td.hostBias[ch];
            const float expectedStd  = std::fabs(td.hostScale[ch]);

            CHECK(std::fabs(mean - expectedMean)               < 1e-4f);
            CHECK(std::fabs(std::sqrt(variance) - expectedStd) < 1e-4f);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch tests — one per supported backend, gated on its compile-time
// availability in mlss-tester.
// ---------------------------------------------------------------------------

TEST_CASE("MVN2: HIP non-relocatable", "[mvn2][hip][gpu]")
{
    if constexpr (mlss::tester::hasHip())
    {
        auto& td = getTestData();
        if (!td.gpuAvailable) SKIP("No compatible GPU detected");

        auto result = runMvn2Pipeline<HipManagedModule, HipDeviceMemory, HipShader>(
            td.bins.data, td.bins.count, /*relocatable=*/false,
            td.hostInput16, td.hostScale16, td.hostBias16);

        REQUIRE(result.size() == td.hostRef.size());
        CHECK(resultsMatch(result, td.hostRef, kTolerance, "HIP vs Host"));
    }
    else
    {
        SKIP("HIP backend not compiled into mlss-tester");
    }
}

TEST_CASE("MVN2: D3D12 relocatable", "[mvn2][d3d][gpu]")
{
    if constexpr (mlss::tester::hasD3D12())
    {
        auto& td = getTestData();
        if (!td.gpuAvailable) SKIP("No compatible GPU detected");

        auto result = runMvn2Pipeline<D3D12ManagedModule, D3D12DeviceMemory, D3D12Shader>(
            td.bins.data, td.bins.count, /*relocatable=*/true,
            td.hostInput16, td.hostScale16, td.hostBias16);

        REQUIRE(result.size() == td.hostRef.size());
        CHECK(resultsMatch(result, td.hostRef, kTolerance, "D3D12 vs Host"));
    }
    else
    {
        SKIP("D3D12 backend not compiled into mlss-tester");
    }
}

// Temporarily disabled: the OpenCL backend currently produces all-zero
// output for tensor arguments, suspected to be an ABI mismatch in how
// double-pointer (storage handle + offset) kernel args are forwarded
// through clSetKernelArg. The same issue is documented in mha_test.cpp.
// HIP, D3D12, and Host backends cover the same logic in the meantime.
// Re-enable once the OpenCL dispatch path is fixed.
#if 0
TEST_CASE("MVN2: OpenCL non-relocatable", "[mvn2][cl][gpu]")
{
    if constexpr (mlss::tester::hasOpenCL())
    {
        auto& td = getTestData();
        if (!td.gpuAvailable) SKIP("No compatible GPU detected");

        auto result = runMvn2Pipeline<ClManagedModule, ClDeviceMemory, ClShader>(
            td.bins.data, td.bins.count, /*relocatable=*/false,
            td.hostInput16, td.hostScale16, td.hostBias16);

        REQUIRE(result.size() == td.hostRef.size());
        CHECK(resultsMatch(result, td.hostRef, kTolerance, "OpenCL vs Host"));
    }
    else
    {
        SKIP("OpenCL backend not compiled into mlss-tester");
    }
}
#endif
