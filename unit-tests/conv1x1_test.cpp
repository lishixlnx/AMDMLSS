/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "test_helpers.hpp"

#include <mlss_tester.hpp>

namespace
{

constexpr MLSSuint32 w = 16;
constexpr MLSSuint32 h = 16;
constexpr MLSSuint32 c = 16;
constexpr MLSSuint32 n = 2;
constexpr MLSSuint32 k = 16;
constexpr MLSSuint32 s = 1;
constexpr MLSSuint32 r = 1;
constexpr float      kTolerance = 5e-2f;

constexpr MLSSuint32 startPadX = 0;
constexpr MLSSuint32 startPadY = 0;
constexpr MLSSuint32 outW      = (w - r + 2 * startPadX) / s + 1;
constexpr MLSSuint32 outH      = (h - r + 2 * startPadY) / s + 1;
constexpr MLSSuint32 endPadX   = 0;
constexpr MLSSuint32 endPadY   = 0;
constexpr MLSSuint32 outPadX   = 0;
constexpr MLSSuint32 outPadY   = 0;
constexpr MLSSuint32 convStrideX   = 1;
constexpr MLSSuint32 convStrideY   = 1;
constexpr MLSSuint32 inputStrideX  = 1;
constexpr MLSSuint32 inputStrideY  = 1;
constexpr MLSSuint32 filterStrideX = 1;
constexpr MLSSuint32 filterStrideY = 1;
constexpr MLSSuint32 groups        = 1;
constexpr MLSSbool   hasBias          = true;
constexpr MLSSbool   crossCorrelation = false;
constexpr MLSSbool   backward         = false;

constexpr MLSSuint32 dNStride = 1;
constexpr MLSSuint32 dHStride = 1;
constexpr MLSSuint32 dCStride = 1;
constexpr MLSSuint32 fKStride = 1;
constexpr MLSSuint32 fCStride = 1;
constexpr MLSSuint32 fRStride = 1;
constexpr MLSSuint32 fSStride = 1;
constexpr MLSSuint32 oNStride = 1;
constexpr MLSSuint32 oHStride = 1;
constexpr MLSSuint32 oKStride = 1;
constexpr MLSSuint32 dOffset  = 1;
constexpr MLSSuint32 oOffset  = 1;
constexpr MLSSuint32 fOffset  = 1;
constexpr MLSSuint32 bOffset  = 1;

constexpr MLSSuint32 activation = MLSS_ACTIVATION_COUNT;
constexpr MLSSuint32 precision  = MLSS_FLOAT16;
constexpr MLSSenum   dataType   = MLSS_FLOAT16;

// ---------------------------------------------------------------------------
// Float16 helpers
// ---------------------------------------------------------------------------
using float16 = __half;

inline float float16ToFloat(float16 value)
{
    return __half2float(value);
}

inline float16 floatToFloat16(float value)
{
    return __float2half(value);
}

void fillDummy(std::vector<float16>& data, float startValue = 0.0f, float step = 1.0f)
{
    float val = startValue;
    for (auto& x : data)
    {
        x = floatToFloat16(val);
        val += step;
    }
}

struct ConvTensors
{
    std::vector<float16> input;
    std::vector<float16> filter;
    std::vector<float16> bias;
    std::vector<float16> output;
};

// ---------------------------------------------------------------------------
// MLSS C API — obtain compiled CONV binaries
// ---------------------------------------------------------------------------
struct MlssBinaries
{
    MLSSbinary* data    = nullptr;
    MLSSsize    count   = 0;
    MLSScontext context = 0;
};

MlssBinaries getConvBinaries(MLSSstring asic)
{
    MlssBinaries out{};
    MLSSstring   opName = const_cast<MLSSstring>(MLSS_CONV);

    CHECK_STATUS(mlssCreateContext(&out.context, asic, opName));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_W, &w));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_H, &h));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_C, &c));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_N, &n));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_K, &k));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_S, &s));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_R, &r));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTW, &outW));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTH, &outH));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_STARTPADX, &startPadX));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_STARTPADY, &startPadY));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ENDPADX, &endPadX));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ENDPADY, &endPadY));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTPADX, &outPadX));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTPADY, &outPadY));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_CONVSTRIDEX, &convStrideX));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_CONVSTRIDEY, &convStrideY));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_INPUTSTRIDEX, &inputStrideX));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_INPUTSTRIDEY, &inputStrideY));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FILTERSTRIDEX, &filterStrideX));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FILTERSTRIDEY, &filterStrideY));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_GROUPS, &groups));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_HASBIAS, &hasBias));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_CROSSCORRELATION, &crossCorrelation));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_BACKWARD, &backward));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DNSTRIDE, &dNStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DHSTRIDE, &dHStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DCSTRIDE, &dCStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FKSTRIDE, &fKStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FCSTRIDE, &fCStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FRSTRIDE, &fRStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FSSTRIDE, &fSStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ONSTRIDE, &oNStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OHSTRIDE, &oHStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OKSTRIDE, &oKStride));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DOFFSET, &dOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OOFFSET, &oOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FOFFSET, &fOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_BOFFSET, &bOffset));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DATATYPE, &dataType));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_PRECISION, &precision));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ACTIVATION, &activation));

    MLSSstatus* pStatuses = nullptr;
    MLSSsize    nStatuses = 0;
    REQUIRE(mlssGetCaps(out.context, &pStatuses, &nStatuses) == MLSS_SUCCESS);

    CHECK_STATUS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;
}

// ---------------------------------------------------------------------------
// Templated GPU dispatch for HIP / D3D12 / OpenCL
// ---------------------------------------------------------------------------
template<typename Module, typename Memory, typename Shader>
std::vector<float> runConvGpu(const MLSSbinary& bin,
                              const ConvTensors& tensor)
{
    ShaderDescriptor desc = buildShaderDescriptor(bin);

    // --- module construction -----------------------------------------------
    [[maybe_unused]] ClContext backendCtx{};
    Module module = [&]() -> Module
    {
        if constexpr (std::is_same_v<Module, ClManagedModule>)
            return Module(backendCtx, desc);
        else
            return Module(desc);
    }();

    REQUIRE(module.isLoaded());

    // --- shader lookup -----------------------------------------------------
    // Backend APIs diverge: HIP/CL return the concrete shader by value;
    // D3D12 returns a `const BaseShaderTag*` that we copy from.
    Shader kernel = [&]() -> Shader
    {
        if constexpr (std::is_same_v<Shader, D3D12Shader>)
        {
            const BaseShaderTag* tag = module.getShader(bin.m_pKernelName);
            REQUIRE(tag != nullptr);
            return *static_cast<const D3D12Shader*>(tag);
        }
        else
        {
            Shader sh = module.getShader(bin.m_pKernelName);
            if constexpr (std::is_same_v<Shader, HipShader>)
            {
                if (!sh.isValid())
                {
                    auto names = module.getShadersNames();
                    if (!names.empty())
                        sh = module.getShader(names[0]);
                }
            }
            return sh;
        }
    }();

    REQUIRE(kernel.isValid());

    // --- memory allocation helper ------------------------------------------
    auto allocMem = [&](std::size_t bytes) -> Memory
    {
        if constexpr (std::is_same_v<Memory, D3D12DeviceMemory>)
            return Memory(module.getDevice()->handle().Get(), bytes);
        else if constexpr (std::is_same_v<Memory, ClDeviceMemory>)
            return Memory(backendCtx, bytes);
        else
            return Memory(bytes);
    };

    Memory devInput  = allocMem(sizeof(float16) * tensor.input.size());
    Memory devFilter = allocMem(sizeof(float16) * tensor.filter.size());
    Memory devBias   = allocMem(sizeof(float16) * tensor.bias.size());
    Memory devOutput = allocMem(sizeof(float16) * tensor.output.size());

    devInput.upload(tensor.input);
    devFilter.upload(tensor.filter);
    if (!tensor.bias.empty()) devBias.upload(tensor.bias);

    devInput.setDoublePointer();
    devFilter.setDoublePointer();
    devBias.setDoublePointer();
    devOutput.setDoublePointer();

    // The Conv1x1 WMMA backend exposes its kernel as a batched GEMM with this mapping:
    //   M       = k                (output channels)
    //   N       = h * w            (output spatial elements)
    //   K       = c                (input channels)
    //   AAddr   = filter           (M x K row-major == [K_out, C_in])
    //   BAddr   = input            (K x N col-major == [H*W, C] row-major == NHWC slice)
    //   CAddr   = bias             (per-M, broadcast over N when broadcastN=1)
    //   DAddr   = output           (M x N row-major == [K_out, H*W] per batch)
    // The filter and bias are shared across the batch, so their batch strides are 0.
    const std::uint32_t gemmM = k;
    const std::uint32_t gemmN = h * w;
    const std::uint32_t gemmK = c;
    const std::uint32_t batchStrideA = 0;
    const std::uint32_t batchStrideB = h * w * c;
    const std::uint32_t batchStrideC = 0;
    const float         alphaVal     = 1.0f;
    const float         betaVal      = hasBias ? 1.0f : 0.0f;
    const std::uint32_t broadcastN   = hasBias ? 1u : 0u;
    const std::uint32_t broadcastM   = 0u;

    std::unordered_map<std::string, KernelArg> argMap;
    argMap.emplace("M",            KernelArg(gemmM));
    argMap.emplace("N",            KernelArg(gemmN));
    argMap.emplace("K",            KernelArg(gemmK));
    argMap.emplace("AAddr",        KernelArg(devFilter.getDoublePointer()));
    argMap.emplace("BAddr",        KernelArg(devInput.getDoublePointer()));
    argMap.emplace("CAddr",        KernelArg(devBias.getDoublePointer()));
    argMap.emplace("DAddr",        KernelArg(devOutput.getDoublePointer()));
    argMap.emplace("batchStrideA", KernelArg(batchStrideA));
    argMap.emplace("batchStrideB", KernelArg(batchStrideB));
    argMap.emplace("batchStrideC", KernelArg(batchStrideC));
    argMap.emplace("alpha",        KernelArg(alphaVal));
    argMap.emplace("beta",         KernelArg(betaVal));
    argMap.emplace("activation",   KernelArg(static_cast<std::uint32_t>(activation)));
    argMap.emplace("param1",       KernelArg(0.0f));
    argMap.emplace("param2",       KernelArg(0.0f));
    argMap.emplace("broadcastN",   KernelArg(broadcastN));
    argMap.emplace("broadcastM",   KernelArg(broadcastM));

    auto args = buildArgsFromBinary(bin, argMap);
    REQUIRE_FALSE(args.empty());

    const dim3 grid (bin.m_grid.m_x,   bin.m_grid.m_y,   bin.m_grid.m_z);
    const dim3 block(bin.m_blocks.m_x, bin.m_blocks.m_y, bin.m_blocks.m_z);

    REQUIRE(kernel.run(args, grid, block));

    std::vector<float16> outFp16(tensor.output.size());
    devOutput.download(outFp16);

    std::vector<float> outFp32(outFp16.size());
    for (std::size_t i = 0; i < outFp16.size(); ++i)
        outFp32[i] = float16ToFloat(outFp16[i]);
    return outFp32;
}

// Reference 1x1 convolution computed in fp32 on the host.
//
// Input  layout: NHWC  (per batch [H*W, C])
// Filter layout: [K_out, C_in]
// Output layout: per batch [K_out, H*W]   (matches the GEMM-style WMMA backend
//                which writes D as M=K_out by N=H*W row-major per batch).
void computeConv1x1NchwOut(const std::vector<float16>& input,
                           const std::vector<float16>& filter,
                           const std::vector<float16>& bias,
                           std::vector<float>&         output,
                           std::uint32_t batch_n,
                           std::uint32_t channels,
                           std::uint32_t spatial,
                           std::uint32_t outChannels,
                           bool          biasEnabled)
{
    const std::size_t outputSize =
        static_cast<std::size_t>(batch_n) * outChannels * spatial;
    output.assign(outputSize, 0.0f);

    for (std::uint32_t b = 0; b < batch_n; ++b)
    {
        for (std::uint32_t outC = 0; outC < outChannels; ++outC)
        {
            const float biasValue = (biasEnabled && !bias.empty())
                ? float16ToFloat(bias[outC]) : 0.0f;

            for (std::uint32_t sp = 0; sp < spatial; ++sp)
            {
                float accumulator = 0.0f;
                for (std::uint32_t inC = 0; inC < channels; ++inC)
                {
                    const std::size_t inputIdx =
                        static_cast<std::size_t>(b) * spatial * channels +
                        static_cast<std::size_t>(sp) * channels + inC;
                    const std::size_t filterIdx =
                        static_cast<std::size_t>(outC) * channels + inC;

                    accumulator += float16ToFloat(input[inputIdx]) *
                                   float16ToFloat(filter[filterIdx]);
                }

                const std::size_t outIdx =
                    static_cast<std::size_t>(b) * outChannels * spatial +
                    static_cast<std::size_t>(outC) * spatial + sp;
                output[outIdx] = accumulator + biasValue;
            }
        }
    }
}

// Shared test data — built once across all TEST_CASEs via static storage.
struct ConvTestData
{
    MlssBinaries               bins{};
    const MLSSbinary*          nonReloc = nullptr;
    const MLSSbinary*          reloc    = nullptr;
    std::vector<float>         hostRef;        // [N, K, H*W] (NCHW)
    ConvTensors                tensors;
    bool                       gpuAvailable = false;
};

ConvTestData& getTestData()
{
    static ConvTestData data = []
    {
        ConvTestData d;

        hipDeviceProp_t props{};
        d.gpuAvailable = (hipGetDeviceProperties(&props, 0) == hipSuccess);
        if (!d.gpuAvailable) return d;

        std::cout << "Detected GPU: " << props.gcnArchName << '\n';

        MLSSstring asic = const_cast<MLSSstring>(MLSS_GFX1200);
        mlssSetVerboseLevel(0);

        d.bins     = getConvBinaries(asic);
        d.nonReloc = findBinary(d.bins.data, d.bins.count, false);
        d.reloc    = findBinary(d.bins.data, d.bins.count, true);

        const std::uint32_t channelsPerGroup = c / groups;

        const std::size_t inputElements  = static_cast<std::size_t>(n) * h * w * c;
        const std::size_t filterElements = static_cast<std::size_t>(k) * r * s * channelsPerGroup;
        const std::size_t biasElements   = hasBias ? static_cast<std::size_t>(k) : 0;
        const std::size_t outputElements = static_cast<std::size_t>(n) * outH * outW * k;

        d.tensors.input.resize(inputElements);
        d.tensors.filter.resize(filterElements);
        d.tensors.bias.resize(biasElements);
        d.tensors.output.resize(outputElements);

        fillDummy(d.tensors.input,  -0.5f, 0.01f);
        fillDummy(d.tensors.filter, 0.1f,  0.005f);
        fillDummy(d.tensors.bias,   0.0f,  0.25f);
        std::fill(d.tensors.output.begin(), d.tensors.output.end(), 0.0f);

        computeConv1x1NchwOut(d.tensors.input, d.tensors.filter, d.tensors.bias,
                              d.hostRef, n, c, h * w, k, hasBias);

        return d;
    }();
    return data;
}

// Element-wise comparison between flat fp32 buffers (used for the NCHW output).
inline bool compareBuffersFlat(const std::vector<float>& a, const std::vector<float>& b,
                               float tolerance, const std::string& label)
{
    if (a.size() != b.size())
    {
        std::cerr << label << ": size mismatch (" << a.size() << " vs " << b.size() << ")\n";
        return false;
    }

    bool pass = true;
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

#if 0
TEST_CASE("Conv1x1: HIP non-relocatable", "[conv1x1][hip]")
{
    if constexpr (mlss::tester::hasHip())
    {
        auto& td = getTestData();
        REQUIRE(td.gpuAvailable);
        REQUIRE(td.nonReloc != nullptr);

        auto result = runConvGpu<HipManagedModule, HipDeviceMemory, HipShader>(
                          *td.nonReloc, td.tensors);
        REQUIRE_FALSE(result.empty());
    }
    else
    {
        SKIP("HIP backend not compiled into mlss-tester");
    }
}


TEST_CASE("Conv1x1: D3D12 relocatable", "[conv1x1][d3d]")
{
    if constexpr (mlss::tester::hasD3D12())
    {
        auto& td = getTestData();
        REQUIRE(td.gpuAvailable);
        REQUIRE(td.reloc != nullptr);

        auto result = runConvGpu<D3D12ManagedModule, D3D12DeviceMemory, D3D12Shader>(
                          *td.reloc, td.tensors);
        REQUIRE_FALSE(result.empty());
        CHECK(compareBuffersFlat(result, td.hostRef, kTolerance, "D3D vs Host"));
    }
    else
    {
        SKIP("D3D12 backend not compiled into mlss-tester");
    }
}

TEST_CASE("Conv1x1: OpenCL non-relocatable", "[conv1x1][cl]")
{
    if constexpr (mlss::tester::hasOpenCL())
    {
        auto& td = getTestData();
        REQUIRE(td.gpuAvailable);
        REQUIRE(td.nonReloc != nullptr);

        auto result = runConvGpu<ClManagedModule, ClDeviceMemory, ClShader>(
                          *td.nonReloc, td.tensors);
        REQUIRE_FALSE(result.empty());
    }
    else
    {
        SKIP("OpenCL backend not compiled into mlss-tester");
    }
}
#else
TEST_CASE("Conv1x1: WIP", "[conv1x1][WIP]")
{
    SKIP("All tests are skipped at this time!");
}
#endif
