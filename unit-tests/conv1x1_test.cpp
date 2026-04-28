/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "test_helpers.hpp"

#include "hip/memory.hpp"
#include "hip/module.hpp"
#include "hip/shader.hpp"
#include <hip/hip_fp16.h>

#ifdef MLSS_D3D_ENABLED
#include "d3d/memory.hpp"
#include "d3d/module.hpp"
#include "d3d/shader.hpp"
#endif

#ifdef MLSS_CL_ENABLED
#include "cl/context.hpp"
#include "cl/memory.hpp"
#include "cl/module.hpp"
#include "cl/shader.hpp"
#endif

namespace
{
// convolution params 
    MLSSuint32 w = 256;                  // Input x-dimension size.
    MLSSuint32 h = 256;                  // Input y-dimension size.
    MLSSuint32 c = 64;                   // Number of channels.
    MLSSuint32 n = 10;                   // Number of batches.
    MLSSuint32 k = 80;                   // Number of features.
    MLSSuint32 s = 1;                    // Filter x-dimension size.
    MLSSuint32 r = 1;                    // Filter y-dimension size.
    MLSSuint32 outW = 254;               // Output x-dimension size.
    MLSSuint32 outH = 254;               // Output y-dimension size.
    MLSSuint32 startPadX = 0;            // Zero padding added to the beginning of the input in the x-dimension.
    MLSSuint32 startPadY = 0;            // Zero padding added to the beginning of the input in the y-dimension.
    MLSSuint32 endPadX = 0;              // Zero padding added to the end of the input in the x-dimension.
    MLSSuint32 endPadY = 0;              // Zero padding added to the end of the input in the y-dimension.
    MLSSuint32 outPadX = 0;              // Zero padding added to the end of the output in the x-dimension.
    MLSSuint32 outPadY = 0;              // Zero padding added to the end of the output in the y-dimension.
    MLSSuint32 convStrideX = 1;          // Step between convolutions in the input x-dimension.
    MLSSuint32 convStrideY = 1;          // Step between convolutions in the input y-dimension.
    MLSSuint32 inputStrideX = 1;         // Step between dot products in the input x-dimension. Adds zero padding in the input.
    MLSSuint32 inputStrideY = 1;         // Step between dot products in the input y-dimension. Adds zero padding in the input.
    MLSSuint32 filterStrideX = 1;        // Step between dot products in the filter x-dimension. Adds zero padding in the filter.
    MLSSuint32 filterStrideY = 1;        // Step between dot products in the filter y-dimension. Adds zero padding in the filter.
    MLSSuint32 groups = 1;               // Split c and k into this many filter groups.
    MLSSbool   hasBias = true;           // If there is a bias tensor.
    MLSSbool   crossCorrelation = false; // If this is a cross correlation instead of a real convolution. Most ML convs are CCs.
    MLSSbool   backward = false;         // If this represents a "backward wrt inputs conv"/"transpose conv"/"deconvolution" layer.
                                         // The above parameters have already been adjusted to convert the backwards conv into an
                                         // equivalent forward conv. The meta command implementation must still swap its C and K
                                         // indices when accessing the filter (see swapCK in GenericConvResources).

    MLSSuint32 dNStride = 1;             // n stride of the input data
    MLSSuint32 dHStride = 1;             // h stride of the input data
    MLSSuint32 dCStride = 1;
    MLSSuint32 fKStride = 1;             // k stride
    MLSSuint32 fCStride = 1;             // c stride
    MLSSuint32 fRStride = 1;             // r stride
    MLSSuint32 fSStride = 1;             // s stride
    MLSSuint32 oNStride = 1;             // n stride of the output data
    MLSSuint32 oHStride = 1;             // h stride of the output data
    MLSSuint32 oKStride = 1;
    MLSSuint32 dOffset  = 1;
    MLSSuint32 oOffset  = 1;
    MLSSuint32 fOffset  = 1;
    MLSSuint32 bOffset  = 1;

    MLSSuint32 activation = MLSS_ACTIVATION_COUNT;
    MLSSuint32 precision  = MLSS_FLOAT16;
    MLSSenum   dataType   = MLSS_FLOAT16;

//==============================================================================
// Conv Tensor populate helper functions 
//==============================================================================
static int computeOutputDim(int input, int padStart, int padEnd, int kernel, int stride)
{
    return (input + padStart + padEnd - kernel) / stride + 1;
}
//==============================================================================
// Float16 Helper Functions
//==============================================================================
using float16 = __half;

inline float16 floatToFloat16(float value)
{
    return __float2half(value);
}

inline float float16ToFloat(float16 value)
{
    return __half2float(value);
}
// Fills `data` with values: start, start+step, start+2*step, ...
// Cast through float to avoid half-precision arithmetic issues.
static void fillDummy(std::vector<float16>& data, float16 startValue = 0.0f, float16 step = 1.0f)
{
    float val = startValue;
    for (auto& x : data)
    {
        x = val;
        val += step;
    }
}
struct ConvTensors
{
    std::vector<float16> input;  // [N][C][H][W]
    std::vector<float16> filter; // [K][R][S][C/groups]
    std::vector<float16> bias;   // [K]
    std::vector<float16> output; // [N][outH][outW][K]
} tensors;

// ---------------------------------------------------------------------------
// MLSS C API — obtain compiled MHA binaries
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
    MLSSstring   opName   = const_cast<MLSSstring>(MLSS_CONV);

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

    MLSSstatus* pStatuses  = nullptr;
    MLSSsize    nStatuses  = 0;
    if (mlssGetCaps(out.context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
    {
        std::cerr << "Conv getCaps failed — configuration not supported on this ASIC\n";
        std::exit(kSkipExitCode);
    }

    CHECK_STATUS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;

    // Populating ConvTensors
    const int channelsPerGroup = c / groups;

    const size_t inputElements  = static_cast<size_t>(n) * h * w * c;
    const size_t filterElements = static_cast<size_t>(k) * r * s * channelsPerGroup;
    const size_t biasElements   = hasBias ? static_cast<size_t>(k) : 0;
    const size_t outputElements = static_cast<size_t>(n) * outH * outW * k;

    tensors.input.resize(inputElements);
    tensors.filter.resize(filterElements);
    tensors.bias.resize(biasElements);
    tensors.output.resize(outputElements);

    // Fill with dummy values
    fillDummy(tensors.input,  1.0f, 1.0f);   // 1.0, 2.0, 3.0, ...
    fillDummy(tensors.filter, 0.1f, 0.1f);   // 0.1, 0.2, 0.3, ...
    fillDummy(tensors.bias,   0.0f, 1.0f);   // 0.0, 1.0, 2.0, ...
    std::fill(tensors.output.begin(), tensors.output.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// Host reference — scaled dot-product attention in fp32
// ---------------------------------------------------------------------------

/*
TensorHost<float> runMhaHost(const TensorHost<float>& Q,
                             const TensorHost<float>& K,
                             const TensorHost<float>& V)
{
    return referenceAttention(Q, K, V, kScale);
}
*/

// ---------------------------------------------------------------------------
// Templated GPU dispatch for HIP / D3D12 / OpenCL
// ---------------------------------------------------------------------------

template<typename Module, typename Memory, typename Shader>
std::vector<float> runConvGpu(const MLSSbinary& bin, const ConvTensors &tensor)
{
    auto desc = buildShaderDescriptor(bin);

    // --- module construction -----------------------------------------------
#ifdef MLSS_CL_ENABLED
    [[maybe_unused]] ClContext backendCtx{};
    Module module = [&]() -> Module
    {
        if constexpr (std::is_same_v<Module, ClManagedModule>)
            return Module(backendCtx, desc);
        else
            return Module(desc);
    }();
#else
    Module module(desc);
#endif

    if (!module.isLoaded())
    {
        std::cerr << "GPU: module failed to load\n";
        return {};
    }

    // --- shader lookup -----------------------------------------------------
    Shader kernel = [&]() -> Shader
    {
        if constexpr (std::is_same_v<Shader, HipShader>)
        {
            Shader sh = module.getShader(bin.m_pKernelName);
            if (!sh.isValid())
            {
                auto names = module.getShadersNames();
/*                if (!names.empty())
                    sh = module.getShader(names[0]);*/
            }
            sh = module.getShader(bin.m_pKernelName);
            return sh;
        }
#ifdef MLSS_CL_ENABLED
        else if constexpr (std::is_same_v<Shader, ClShader>)
        {
            return module.getShader(bin.m_pKernelName);
        }
#endif
        else
        {
            return module.getShader(bin.m_pKernelName);
        }
    }();

    if (!kernel.isValid())
    {
        std::cerr << "GPU: kernel '" << bin.m_pKernelName << "' not found\n";
        return {};
    }

    // --- memory allocation helper ------------------------------------------
    constexpr std::size_t fp16Bytes = sizeof(float16_t);

    auto allocMem = [&](std::size_t bytes) -> Memory
    {
        (void)bytes;
#ifdef MLSS_D3D_ENABLED
        if constexpr (std::is_same_v<Memory, D3D12DeviceMemory>)
            return Memory(module.getDevice()->handle().Get(), bytes);
        else
#endif
        {
#ifdef MLSS_CL_ENABLED
            if constexpr (std::is_same_v<Memory, ClDeviceMemory>)
                return Memory(backendCtx, bytes);
            else
#endif
                return Memory(bytes);
        }
    };

    Memory devInput  = allocMem(sizeof(float16) * tensor.input.size());
    Memory devFilter = allocMem(sizeof(float16) * tensor.filter.size());
    Memory devBias   = allocMem(sizeof(float16) * tensor.bias.size());
    Memory devOutput = allocMem(sizeof(float16) * tensor.output.size());

    devInput.upload(tensor.input);
    devFilter.upload(tensor.filter);
    if(!tensor.bias.empty()) devBias.upload(tensor.bias);

    devInput.setDoublePointer();
    devFilter.setDoublePointer();
    devBias.setDoublePointer();
    devOutput.setDoublePointer();

    std::unordered_map<std::string, KernelArg> argMap;

    // TODO(ram): move this to a helper function
    // ----- Tensor pointers -------------------------------------------------
    argMap.emplace("dataAddr",   KernelArg(devInput.getDoublePointer()));
    argMap.emplace("filterAddr", KernelArg(devFilter.getDoublePointer()));
    argMap.emplace("biasAddr",   KernelArg(devBias.getDoublePointer()));
    argMap.emplace("outputAddr", KernelArg(devOutput.getDoublePointer()));

    // ----- Shape / conv params (taken straight from the globals) -----------
    argMap.emplace("n",    KernelArg(n));
    argMap.emplace("c",    KernelArg(c));
    argMap.emplace("h",    KernelArg(h));
    argMap.emplace("w",    KernelArg(w));
    argMap.emplace("k",    KernelArg(k));
    argMap.emplace("r",    KernelArg(r));
    argMap.emplace("s",    KernelArg(s));
    argMap.emplace("outH", KernelArg(outH));
    argMap.emplace("outW", KernelArg(outW));

    // Shader names map to startPadY/startPadX and want INT32, not UINT32.
    argMap.emplace("padH", KernelArg(static_cast<std::int32_t>(startPadY)));
    argMap.emplace("padW", KernelArg(static_cast<std::int32_t>(startPadX)));

    // Group count: the shader uses "G" (group index extent) and "nGroups"
    // (workgroup multiplier). With no batched grouping, both equal `groups`.
    argMap.emplace("G",       KernelArg(groups));
    argMap.emplace("nGroups", KernelArg(groups));
    // ----- Offsets ---------------------------------------------------------
    argMap.emplace("dOffset", KernelArg(static_cast<std::uint64_t>(dOffset)));
    argMap.emplace("fOffset", KernelArg(static_cast<std::uint64_t>(fOffset)));
    argMap.emplace("oOffset", KernelArg(static_cast<std::uint64_t>(oOffset)));
    argMap.emplace("bOffset", KernelArg(static_cast<std::uint64_t>(bOffset)));

    // ----- Strides ---------------------------------------------------------
    argMap.emplace("dNStride", KernelArg(dNStride));
    argMap.emplace("dCStride", KernelArg(dCStride));
    argMap.emplace("dHStride", KernelArg(dHStride));
    argMap.emplace("fKStride", KernelArg(fKStride));
    argMap.emplace("fCStride", KernelArg(fCStride));
    argMap.emplace("fRStride", KernelArg(fRStride));
    argMap.emplace("oNStride", KernelArg(oNStride));
    argMap.emplace("oKStride", KernelArg(oKStride));
    argMap.emplace("oHStride", KernelArg(oHStride));

    // Group strides — no equivalent in the conv params; leave 0 for groups==1.
    argMap.emplace("dGStride", KernelArg(std::uint32_t{0}));
    argMap.emplace("fGStride", KernelArg(std::uint32_t{0}));
    argMap.emplace("oGStride", KernelArg(std::uint32_t{0}));

    // ----- Flags / scaling / activation ------------------------------------
    // flags64 is a packed shader-control word (e.g. has-bias bit). Derive
    // from the existing params where possible instead of inventing new ones.
    const std::uint64_t flags64 = (hasBias ? 1ull : 0ull);
    argMap.emplace("flags64",        KernelArg(flags64));
    argMap.emplace("alpha",          KernelArg(1.0f));
    argMap.emplace("beta",           KernelArg(0.0f));
    argMap.emplace("activationMode", KernelArg(static_cast<std::uint8_t>(activation)));

    // TODO(ram): What are these?
    // ----- Reserved padding slots in the arg list --------------------------
    argMap.emplace("reserved3", KernelArg(std::uint64_t{0}));
    argMap.emplace("reserved4", KernelArg(std::uint32_t{0}));
    argMap.emplace("reserved5", KernelArg(std::uint32_t{0}));
    argMap.emplace("reserved6", KernelArg(std::uint32_t{0}));
    argMap.emplace("reserved7", KernelArg(std::uint8_t{0}));
    argMap.emplace("reserved8", KernelArg(std::uint16_t{0}));

    auto args = buildArgsFromBinary(bin, argMap);
    if (args.empty())
    {
        std::cerr << "GPU: failed to build kernel arguments from m_argList\n";
        return {};
    }

    const dim3 grid (bin.m_grid.m_x,   bin.m_grid.m_y,   bin.m_grid.m_z);
    const dim3 block(bin.m_blocks.m_x, bin.m_blocks.m_y, bin.m_blocks.m_z);

    if (!kernel.run(args, grid, block))
    {
        std::cerr << "GPU: kernel launch failed\n";
        return {};
    }

    std::vector<float16> outFp16(tensor.output.size());
    devOutput.download(outFp16);

    std::vector<float> outFp32(outFp16.size());
    for (std::size_t i = 0; i < outFp16.size(); ++i)
        outFp32[i] = float16ToFloat(outFp16[i]);
    return outFp32;
}

} // namespace

int main()
{
    try
    {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        {
            std::cerr << "SKIP: cannot query GPU properties\n";
            return kSkipExitCode;
        }
        std::cout << "Detected GPU: " << props.gcnArchName << "\n\n";

        MLSSstring asic = const_cast<MLSSstring>(MLSS_GFX1201);
        CHECK_STATUS(mlssSetVerboseLevel(0));

        auto convBins = getConvBinaries(asic);
        const MLSSbinary* nonReloc = findBinary(convBins.data, convBins.count, false);
        const MLSSbinary* reloc = findBinary(convBins.data, convBins.count, true);
        if (nonReloc == nullptr && reloc == nullptr)
        {
            std::cerr << "FAIL: no usable binary found\n";
            return EXIT_FAILURE;
        }

        bool allPassed = true;
        // --- HIP (non-relocatable) -----------------------------------------
        if (nonReloc != nullptr)
        {
            std::cout << "\n--- HIP (non-relocatable) ---\n";
            auto resultVec = runConvGpu<HipManagedModule, HipDeviceMemory, HipShader>(
                *nonReloc, tensors);
            if (resultVec.empty())
            {
                std::cerr << "FAIL: HIP MHA kernel produced no results\n";
                allPassed = false;
            }
/*            else if (!compareBuffers(transposeHeadSeq(vectorToTensor(resultVec, gpuOutShape)),
                                     hostRef, kTolerance, "HIP vs Host"))
            {
                allPassed = false;
            }*/
/*            else
            {
                std::cout << "PASS: HIP vs Host (tolerance=" << kTolerance << ")\n";
            }*/
        }

    }
    /*
        // --- D3D12 (relocatable) -------------------------------------------
#ifdef MLSS_D3D_ENABLED
        if (reloc != nullptr)
        {
            std::cout << "\n--- D3D12 (relocatable) ---\n";
            auto resultVec = runMhaGpu<D3D12ManagedModule, D3D12DeviceMemory, D3D12Shader>(
                                 *reloc, hostQ16, hostK16, hostV16);
            if (resultVec.empty())
            {
                std::cerr << "FAIL: D3D MHA kernel produced no results\n";
                allPassed = false;
            }
            else if (!compareBuffers(transposeHeadSeq(vectorToTensor(resultVec, gpuOutShape)),
                                     hostRef, kTolerance, "D3D vs Host"))
            {
                allPassed = false;
            }
            else
            {
                std::cout << "PASS: D3D vs Host (tolerance=" << kTolerance << ")\n";
            }
        }
#endif

        // --- OpenCL (non-relocatable) --------------------------------------
#ifdef MLSS_CL_ENABLED
        if (nonReloc != nullptr)
        {
            std::cout << "\n--- OpenCL (non-relocatable) ---\n";
            auto resultVec = runMhaGpu<ClManagedModule, ClDeviceMemory, ClShader>(
                                 *nonReloc, hostQ16, hostK16, hostV16);
            if (resultVec.empty())
            {
                std::cerr << "WARN: CL MHA kernel produced no results\n";
            }
            else if (!compareBuffers(transposeHeadSeq(vectorToTensor(resultVec, gpuOutShape)),
                                     hostRef, kTolerance, "CL vs Host"))
            {
                allPassed = false;
            }
            else
            {
                std::cout << "PASS: CL vs Host (tolerance=" << kTolerance << ")\n";
            }
        }
#endif

        std::cout << "\n=== MHA Test " << (allPassed ? "PASSED" : "FAILED") << " ===\n";
        return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "FATAL: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    */
    catch (...)
    {
        std::cerr << "FATAL: unknown exception\n";
        return EXIT_FAILURE;
    }
}
