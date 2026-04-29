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

static_assert(mlss::tester::hasHip(),     "Conv unit tests require the HIP backend");
static_assert(mlss::tester::hasD3D12(),   "Conv unit tests require the D3D12 backend");
static_assert(mlss::tester::hasOpenCL(),  "Conv unit tests require the OpenCL backend");

#define MLSS_CL_ENABLED  0;
#define MLSS_HIP_ENABLED 0;

namespace
{

    const MLSSuint32 w = 256;                  // Input x-dimension size.
    const MLSSuint32 h = 256;                  // Input y-dimension size.
    const MLSSuint32 c = 64;                   // Number of channels.
    const MLSSuint32 n = 10;                   // Number of batches.
    const MLSSuint32 k = 80;                   // Number of features.
    const MLSSuint32 s = 1;                    // Filter x-dimension size.
    const MLSSuint32 r = 1;                    // Filter y-dimension size.

/*    const MLSSuint32 w = 20;                  // Input x-dimension size.
    const MLSSuint32 h = 20;                  // Input y-dimension size.
    const MLSSuint32 c = 10;                   // Number of channels.
    const MLSSuint32 n = 1;                   // Number of batches.
    const MLSSuint32 k = 2;                   // Number of features.
    const MLSSuint32 s = 1;                    // Filter x-dimension size.
    const MLSSuint32 r = 1;                    // Filter y-dimension size.*/
    const MLSSuint32 startPadX = 0;            // Zero padding added to the beginning of the input in the x-dimension.
    const MLSSuint32 startPadY = 0;            // Zero padding added to the beginning of the input in the y-dimension.
    const MLSSuint32 outW = (w - r + 2 * startPadX) / s + 1;               // Output x-dimension size.
    const MLSSuint32 outH = (h - r + 2 * startPadY) / s + 1;               // Output y-dimension size.
    const MLSSuint32 endPadX = 0;              // Zero padding added to the end of the input in the x-dimension.
    const MLSSuint32 endPadY = 0;              // Zero padding added to the end of the input in the y-dimension.
    const MLSSuint32 outPadX = 0;              // Zero padding added to the end of the output in the x-dimension.
    const MLSSuint32 outPadY = 0;              // Zero padding added to the end of the output in the y-dimension.
    const MLSSuint32 convStrideX = 1;          // Step between convolutions in the input x-dimension.
    const MLSSuint32 convStrideY = 1;          // Step between convolutions in the input y-dimension.
    const MLSSuint32 inputStrideX = 1;         // Step between dot products in the input x-dimension. Adds zero padding in the input.
    const MLSSuint32 inputStrideY = 1;         // Step between dot products in the input y-dimension. Adds zero padding in the input.
    const MLSSuint32 filterStrideX = 1;        // Step between dot products in the filter x-dimension. Adds zero padding in the filter.
    const MLSSuint32 filterStrideY = 1;        // Step between dot products in the filter y-dimension. Adds zero padding in the filter.
    const MLSSuint32 groups = 1;               // Split c and k into this many filter groups.
    const MLSSbool   hasBias = true;           // If there is a bias tensor.
    const MLSSbool   crossCorrelation = false; // If this is a cross correlation instead of a real convolution. Most ML convs are CCs.
    const MLSSbool   backward = false;         // If this represents a "backward wrt inputs conv"/"transpose conv"/"deconvolution" layer.
                                         // The above parameters have already been adjusted to convert the backwards conv into an
                                         // equivalent forward conv. The meta command implementation must still swap its C and K
                                         // indices when accessing the filter (see swapCK in GenericConvResources).

    const MLSSuint32 dNStride = 1;             // n stride of the input data
    const MLSSuint32 dHStride = 1;             // h stride of the input data
    const MLSSuint32 dCStride = 1;
    const MLSSuint32 fKStride = 1;             // k stride
    const MLSSuint32 fCStride = 1;             // c stride
    const MLSSuint32 fRStride = 1;             // r stride
    const MLSSuint32 fSStride = 1;             // s stride
    const MLSSuint32 oNStride = 1;             // n stride of the output data
    const MLSSuint32 oHStride = 1;             // h stride of the output data
    const MLSSuint32 oKStride = 1;
    const MLSSuint32 dOffset  = 1;
    const MLSSuint32 oOffset  = 1;
    const MLSSuint32 fOffset  = 1;
    const MLSSuint32 bOffset  = 1;

    const MLSSuint32 activation = MLSS_ACTIVATION_COUNT;
    const MLSSuint32 precision  = MLSS_FLOAT16;
    const MLSSenum   dataType   = MLSS_FLOAT16;

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
inline float float16ToFloat(float16 value)
{
    return __half2float(value);
}

inline float16 floatToFloat16(float value)
{
    return __float2half(value);
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
}

// ---------------------------------------------------------------------------
// Templated GPU dispatch for HIP / D3D12 / OpenCL
// ---------------------------------------------------------------------------
template<typename Module, typename Memory, typename Shader>
std::vector<float> runConvGpu(const MLSSbinary& bin, 
                              const ConvTensors &tensor)
{
    ShaderDescriptor desc = buildShaderDescriptor(bin);

    // --- module construction -----------------------------------------------
#if MLSS_CL_ENABLED
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
                if (!names.empty())
                    sh = module.getShader(names[0]);
            }
            sh = module.getShader(bin.m_pKernelName);
            return sh;
        }
#if MLSS_CL_ENABLED
        else if constexpr (std::is_same_v<Shader, ClShader>)
        {
            return module.getShader(bin.m_pKernelName);
        }
#endif
#if MLSS_D3D_ENABLED
        else if constexpr (std::is_same_v<Shader, D3D12Shader>)
        {
            // D3D12ManagedModule::getShader returns const BaseShaderTag*.
            // Downcast to the concrete D3D12Shader and copy it out by value.
            const BaseShaderTag* tag = module.getShader(bin.m_pKernelName);
            if (tag == nullptr)
                return Shader{};
            return *static_cast<const D3D12Shader*>(tag);
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
#if MLSS_CL_ENABLED
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

    // this is for winograds
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
    // return std::vector<float>();
}

// Shared test data — computed once across all SECTIONs via static storage
struct ConvTestData
{
    MlssBinaries           bins{};
    const MLSSbinary*      nonReloc = nullptr;
    const MLSSbinary*      reloc    = nullptr;
    TensorHost<float>      hostRef;
    ConvTensors            tensors;
    std::vector<std::uint32_t> gpuOutShape;
    bool                   gpuAvailable = false;
};


void computeConvCpu(
    const std::vector<float16>& input,
    const std::vector<float16>& filter,
    const std::vector<float16>& bias,
    TensorHost<float> &result,
    uint32_t n,          // batch size
    uint32_t c,          // input channels
    uint32_t h,          // input height
    uint32_t w,          // input width
    uint32_t k,          // output channels
    uint32_t groups,
    bool hasBias,
    uint32_t outH,
    uint32_t outW,
    uint32_t startPadY = 0,
    uint32_t startPadX = 0,
    uint32_t endPadY = 0,
    uint32_t endPadX = 0,
    uint32_t convStrideY = 1,
    uint32_t convStrideX = 1)
{
    std::vector<float> output(n * outH * outW * k, 0.0f);
    
    uint32_t channelsPerGroup = c / groups;
    uint32_t filtersPerGroup = k / groups;
    
    // For each batch
    for (uint32_t batch = 0; batch < n; ++batch)
    {
        // For each output position
        for (uint32_t oh = 0; oh < outH; ++oh)
        {
            for (uint32_t ow = 0; ow < outW; ++ow)
            {
                // Input coordinates (accounting for padding and stride)
                int32_t ih = static_cast<int32_t>(oh * convStrideY) - static_cast<int32_t>(startPadY);
                int32_t iw = static_cast<int32_t>(ow * convStrideX) - static_cast<int32_t>(startPadX);
                
                // For each output channel
                for (uint32_t out_c = 0; out_c < k; ++out_c)
                {
                    float accumulator = 0.0f;
                    
                    // Determine which group this output channel belongs to
                    uint32_t group_id = out_c / filtersPerGroup;
                    uint32_t filter_idx_in_group = out_c % filtersPerGroup;
                    
                    // For a 1x1 convolution, we only have one position (0, 0)
                    // and we sum over all input channels in this group
                    for (uint32_t in_c = 0; in_c < channelsPerGroup; ++in_c)
                    {
                        // Skip if out of bounds
                        if (ih < 0 || ih >= static_cast<int32_t>(h) ||
                            iw < 0 || iw >= static_cast<int32_t>(w))
                        {
                            continue;  // Zero padding
                        }
                        
                        // Input index: [N][H][W][C] layout
                        uint32_t input_idx = batch * (h * w * c) +
                                           static_cast<uint32_t>(ih) * (w * c) +
                                           static_cast<uint32_t>(iw) * c +
                                           (group_id * channelsPerGroup + in_c);
                        
                        // Filter index: [K][C/groups] layout
                        uint32_t filter_idx = out_c * channelsPerGroup + in_c;
                        
                        float in_val = float16ToFloat(input[input_idx]);
                        float filt_val = float16ToFloat(filter[filter_idx]);
                        accumulator += in_val * filt_val;
                    }
                    
                    // Add bias if present
                    if (hasBias && !bias.empty())
                    {
                        accumulator += float16ToFloat(bias[out_c]);
                    }
                    
                    // Output index: [N][outH][outW][K] layout
                    uint32_t output_idx = batch * (outH * outW * k) +
                                        oh * (outW * k) +
                                        ow * k +
                                        out_c;
                    
                    output[output_idx] = accumulator;
                }
            }
        }
    }
    
    std::memcpy(result.data(), output.data(), output.size() * sizeof(float));
}

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

        // Populating ConvTensors
        const int channelsPerGroup = c / groups;

        const size_t inputElements  = static_cast<size_t>(n) * h * w * c;
        const size_t filterElements = static_cast<size_t>(k) * r * s * channelsPerGroup;
        const size_t biasElements   = hasBias ? static_cast<size_t>(k) : 0;
        const size_t outputElements = static_cast<size_t>(n) * outH * outW * k;

        d.tensors.input.resize(inputElements);
        d.tensors.filter.resize(filterElements);
        d.tensors.bias.resize(biasElements);
        d.tensors.output.resize(outputElements);

        // Fill with dummy values
        fillDummy(d.tensors.input,  1.0f, 1.0f);   // 1.0, 2.0, 3.0, ...
        fillDummy(d.tensors.filter, 0.1f, 0.1f);   // 0.1, 0.2, 0.3, ...
        fillDummy(d.tensors.bias,   0.0f, 1.0f);   // 0.0, 1.0, 2.0, ...
        std::fill(d.tensors.output.begin(), d.tensors.output.end(), 0.0f);


        std::vector<std::uint32_t> shape = {n, outH, outW, k};
        d.hostRef = TensorHost<float>(shape);  // pre-allocate
/*        computeConvCpu(d.tensors.input, d.tensors.filter, d.tensors.bias, d.hostRef, 
                                      n, c, h, w, k, groups, hasBias, outH, outW, 
                                      startPadY, startPadX, endPadY, endPadX, 
                                      convStrideY, convStrideX);
            
        d.gpuOutShape = {n, outH, outW, k};*/

        return d;
    }();
    return data;
}

} // namespace

TEST_CASE("MHA: D3D12 relocatable", "[mha][d3d]")
{
    ConvTestData& td = getTestData();

    REQUIRE(td.gpuAvailable);
    REQUIRE(td.reloc != nullptr);

    auto result = runConvGpu<D3D12ManagedModule, D3D12DeviceMemory, D3D12Shader>(
                  td.bins.data[0], td.tensors);
    // REQUIRE_FALSE(result.empty());
    // CHECK(compareBuffers(transposeHeadSeq(vectorToTensor(result, td.gpuOutShape)),
    //                     td.hostRef, kTolerance, "D3D vs Host"));
}

