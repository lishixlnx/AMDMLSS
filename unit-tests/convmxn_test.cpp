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

#define MLSS_CL_ENABLED  0
#define MLSS_D3D_ENABLED 0 
#define MLSS_HIP_ENABLED 1 

namespace
{
    const MLSSuint32 w = 112;                  // Input x-dimension size.
    const MLSSuint32 h = 112;                  // Input y-dimension size.
    const MLSSuint32 c = 64;                   // Number of channels.
    const MLSSuint32 n = 1;                    // Number of batches.
    const MLSSuint32 k = 128;                  // Number of features.
    const MLSSuint32 s = 3;                    // Filter x-dimension size.
    const MLSSuint32 r = 3;                    // Filter y-dimension size.
    const MLSSuint32 startPadX = 1;            // Zero padding added to the beginning of the input in the x-dimension.
    const MLSSuint32 startPadY = 1;            // Zero padding added to the beginning of the input in the y-dimension.
    const MLSSuint32 endPadX = 1;              // Zero padding added to the end of the input in the x-dimension.
    const MLSSuint32 endPadY = 1;              // Zero padding added to the end of the input in the y-dimension.
    const MLSSuint32 outPadX = 0;              // Zero padding added to the end of the output in the x-dimension.
    const MLSSuint32 outPadY = 0;              // Zero padding added to the end of the output in the y-dimension.
    const MLSSuint32 convStrideX = 1;          // Step between convolutions in the input x-dimension.
    const MLSSuint32 convStrideY = 1;          // Step between convolutions in the input y-dimension.
    // Standard conv output formula: (input + padStart + padEnd - kernel) / convStride + 1.
    // The previous version divided by `s` (filter width) instead of `convStrideX`,
    // which under-sized output buffers by a factor of (convStride / s) and caused
    // the kernel to write past `devOutput`, leaving the downloaded buffer all zero.
    const MLSSuint32 outW = 112; // Output x-dimension size.
    const MLSSuint32 outH = 112; // Output y-dimension size.
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

    const MLSSuint32 dNStride = groups * (c / groups) * h * w;        // n stride of the input data (elements)
    const MLSSuint32 dHStride = w;                                    // h stride of the input data
    const MLSSuint32 dCStride = h * w;                                // c stride of the input data
    const MLSSuint32 fKStride = (c / groups) * r * s;                 // k stride of the filter
    const MLSSuint32 fCStride = r * s;                                // c stride of the filter
    const MLSSuint32 fRStride = s;                                    // r stride of the filter
    const MLSSuint32 fSStride = 1;                                    // s stride of the filter
    const MLSSuint32 oNStride = groups * (k / groups) * outH * outW;  // n stride of the output data
    const MLSSuint32 oHStride = outW;                                 // h stride of the output data
    const MLSSuint32 oKStride = outH * outW;                          // k stride of the output data
    const MLSSuint32 dOffset  = 0;
    const MLSSuint32 oOffset  = 0;
    const MLSSuint32 fOffset  = 0;
    const MLSSuint32 bOffset  = 0;

    const MLSSuint32 activation = MLSS_ACTIVATION_RELU;
    /*const MLSSuint32 precision  = MLSS_FLOAT16; */
    const MLSSuint32 precision  = MLSSPrecisionFlag::MLSS_PRECISION_FLOAT16_ADD_FLOAT32; 
    const MLSSenum   dataType   = MLSS_FLOAT32;

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
    // NOTE: tensor allocation/population lives in getTestData(), which owns
    // the per-test ConvTensors instance. Do not duplicate it here.
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

    // Binary 'NAVI31_fp32_f2x3_stride1' uses fp32 tensors. Convert the fp16
    // host tensors to fp32 before uploading, and allocate fp32 device buffers.
    auto toFp32 = [](const std::vector<float16>& src)
    {
        std::vector<float> dst(src.size());
        for (std::size_t i = 0; i < src.size(); ++i)
            dst[i] = float16ToFloat(src[i]);
        return dst;
    };

    const std::vector<float> inputFp32  = toFp32(tensor.input);
    const std::vector<float> filterFp32 = toFp32(tensor.filter);
    const std::vector<float> biasFp32   = toFp32(tensor.bias);

    Memory devInput  = allocMem(sizeof(float) * tensor.input.size());
    Memory devFilter = allocMem(sizeof(float) * tensor.filter.size());
    Memory devBias   = allocMem(sizeof(float) * tensor.bias.size());
    Memory devOutput = allocMem(sizeof(float) * tensor.output.size());

    devInput.upload(inputFp32);
    devFilter.upload(filterFp32);
    if (!biasFp32.empty()) devBias.upload(biasFp32);

    /*    devInput.setDoublePointer();
        devFilter.setDoublePointer();
        devBias.setDoublePointer();
        devOutput.setDoublePointer();*/

    // --- extra device buffers required by the kernel's full ABI -----------
    // syncAddr is an inter-workgroup sync counter array (one u32 per workgroup).
    // accAddr is a per-output-element fp32 accumulator scratch buffer.
    // Both must be zero-initialised before launch.
    // nGroups (the workgroup multiplier along K) is chosen by the producer
    // based on the shape + macroTile. It always equals grid.x / (N * groups)
    // because the launch grid is N*G*nGroups workgroups.
    const std::uint32_t kNGroups =
        bin.m_grid.m_x / (static_cast<std::uint32_t>(n) * static_cast<std::uint32_t>(groups));
    const std::size_t syncBufBytes = static_cast<std::size_t>(kNGroups) * sizeof(std::uint32_t);
    const std::size_t accBufBytes  = sizeof(float) * tensor.output.size();

    Memory devSync = allocMem(syncBufBytes);
    Memory devAcc  = allocMem(accBufBytes);

    std::vector<std::uint8_t> zerosSync(syncBufBytes, 0);
    std::vector<std::uint8_t> zerosAcc(accBufBytes,  0);
    devSync.upload(zerosSync);
    devAcc.upload(zerosAcc);

    // Capture the device pointers BEFORE storing in argMap
    void* inPtr   = reinterpret_cast<void*>(devInput.data());
    void* filtPtr = reinterpret_cast<void*>(devFilter.data());
    void* biasPtr = reinterpret_cast<void*>(devBias.data());
    void* outPtr  = reinterpret_cast<void*>(devOutput.data());
    void* syncPtr = reinterpret_cast<void*>(devSync.data());
    void* accPtr  = reinterpret_cast<void*>(devAcc.data());

    // flags64 bit layout (see hipKernelLaunch.cpp / conv_base.hpp):
    //   bit  7 = F_BIAS
    //   bit  9 = F_NKCHR_STRIDES         (deprecated, recommended=1)
    //   bit 14 = F_USE_ACTIVATION_MODE   (deprecated, recommended=1)
    //   bit 15 = F_USE_EXTENDED_FLAGS_64 (deprecated, recommended=1)
    const std::uint64_t flags64 =
          (hasBias ? (1ull << 7) : 0ull)
        | (1ull << 9)
        | (1ull << 14)
        | (1ull << 15);

    std::unordered_map<std::string, KernelArg> argMap;

    // Schema source of truth: winograd_conv_ARGS_CONSTANTS in
    //   modules/shaders/src/operators/impl/conv/mxn/Winograd/Base/shadersConstants.hpp
    // Names and types below MUST match that array exactly. Order here is for
    // readability only; buildArgsFromBinary reorders by m_place.

    // ----- Shape / counts (places 0..5) ------------------------------------
    argMap.emplace("n",              KernelArg(static_cast<std::uint32_t>(n)));        //  0  u32
    argMap.emplace("c",              KernelArg(static_cast<std::uint32_t>(c)));        //  1  u32
    argMap.emplace("h",              KernelArg(static_cast<std::uint32_t>(h)));        //  2  u32
    argMap.emplace("w",              KernelArg(static_cast<std::uint32_t>(w)));        //  3  u32
    argMap.emplace("k",              KernelArg(static_cast<std::uint32_t>(k)));        //  4  u32
    argMap.emplace("nGroups",        KernelArg(kNGroups));                             //  5  u32 (workgroup multiplier, NOT conv groups)

    // ----- Flags + tensor pointers (places 6..10) --------------------------
    argMap.emplace("flags64",        KernelArg(flags64));                              //  6  u64
    argMap.emplace("dataAddr",       KernelArg(inPtr));                                //  7  u64 (ptr)
    argMap.emplace("filterAddr",     KernelArg(filtPtr));                              //  8  u64 (ptr)
    argMap.emplace("outputAddr",     KernelArg(outPtr));                               //  9  u64 (ptr)
    argMap.emplace("reserved3",      KernelArg(std::uint64_t{0}));                     // 10  u64

    // ----- Conv geometry (places 11..16) -----------------------------------
    argMap.emplace("r",              KernelArg(static_cast<std::uint32_t>(r)));        // 11  u32
    argMap.emplace("s",              KernelArg(static_cast<std::uint32_t>(s)));        // 12  u32
    argMap.emplace("padH",           KernelArg(static_cast<std::int32_t>(startPadY))); // 13  i32
    argMap.emplace("padW",           KernelArg(static_cast<std::int32_t>(startPadX))); // 14  i32
    argMap.emplace("outH",           KernelArg(static_cast<std::uint32_t>(outH)));     // 15  u32
    argMap.emplace("outW",           KernelArg(static_cast<std::uint32_t>(outW)));     // 16  u32

    // ----- Bias + scaling (places 17..19) ----------------------------------
    argMap.emplace("biasAddr",       KernelArg(biasPtr));                              // 17  u64 (ptr)
    argMap.emplace("alpha",          KernelArg(1.0f));                                 // 18  f32
    argMap.emplace("beta",           KernelArg(0.0f));                                 // 19  f32

    // ----- Offsets (places 20..23) -----------------------------------------
    argMap.emplace("dOffset",        KernelArg(static_cast<std::uint64_t>(dOffset))); // 20  u64
    argMap.emplace("fOffset",        KernelArg(static_cast<std::uint64_t>(fOffset))); // 21  u64
    argMap.emplace("oOffset",        KernelArg(static_cast<std::uint64_t>(oOffset))); // 22  u64
    argMap.emplace("bOffset",        KernelArg(static_cast<std::uint64_t>(bOffset))); // 23  u64

    // ----- Data strides (places 24..27) ------------------------------------
    argMap.emplace("dNStride",       KernelArg(static_cast<std::uint32_t>(dNStride))); // 24  u32
    argMap.emplace("dCStride",       KernelArg(static_cast<std::uint32_t>(dCStride))); // 25  u32
    argMap.emplace("dHStride",       KernelArg(static_cast<std::uint32_t>(dHStride))); // 26  u32
    argMap.emplace("reserved4",      KernelArg(std::uint32_t{0}));                     // 27  u32

    // ----- Filter strides (places 28..31) ----------------------------------
    argMap.emplace("fKStride",       KernelArg(static_cast<std::uint32_t>(fKStride))); // 28  u32
    argMap.emplace("fCStride",       KernelArg(static_cast<std::uint32_t>(fCStride))); // 29  u32
    argMap.emplace("fRStride",       KernelArg(static_cast<std::uint32_t>(fRStride))); // 30  u32
    argMap.emplace("reserved5",      KernelArg(std::uint32_t{0}));                     // 31  u32

    // ----- Output strides (places 32..35) ----------------------------------
    argMap.emplace("oNStride",       KernelArg(static_cast<std::uint32_t>(oNStride))); // 32  u32
    argMap.emplace("oKStride",       KernelArg(static_cast<std::uint32_t>(oKStride))); // 33  u32
    argMap.emplace("oHStride",       KernelArg(static_cast<std::uint32_t>(oHStride))); // 34  u32
    argMap.emplace("reserved6",      KernelArg(std::uint32_t{0}));                     // 35  u32

    // ----- Group count + group strides (places 36..39) ---------------------
    argMap.emplace("G",              KernelArg(static_cast<std::uint32_t>(groups)));   // 36  u32
    argMap.emplace("dGStride",       KernelArg(std::uint32_t{0}));                     // 37  u32
    argMap.emplace("fGStride",       KernelArg(std::uint32_t{0}));                     // 38  u32
    argMap.emplace("oGStride",       KernelArg(std::uint32_t{0}));                     // 39  u32

    // ----- Activation + sync control (places 40..44) -----------------------
    // NOTE: The kernel's activation_mode is *not* the MLSS enum. It uses the
    // MIOpen-style internal codes (RELU=4). MLSS_ACTIVATION_RELU = 9, which the
    // kernel would misinterpret as a different activation. Hard-code 4 for now.
    constexpr std::uint8_t kKernelActivationRelu = 4;
    argMap.emplace("activationMode", KernelArg(kKernelActivationRelu));                 // 40  u8  offset 200
    argMap.emplace("syncLimit",      KernelArg(static_cast<std::uint8_t>(255)));        // 41  u8  offset 201 (DEFAULT_SYNC_LIMIT)
    argMap.emplace("syncPeriod",     KernelArg(static_cast<std::uint8_t>(0)));          // 42  u8  offset 202
    argMap.emplace("reserved8",      KernelArg(static_cast<std::uint8_t>(0)));          // 43  u8  offset 203
    argMap.emplace("reserved9",      KernelArg(static_cast<std::uint32_t>(0)));         // 44  u32 offset 204

    // ----- Sync / accum buffers + a_offset (places 45..47) -----------------
    argMap.emplace("syncAddr",       KernelArg(syncPtr));                               // 45  u64 (ptr) offset 208
    argMap.emplace("accAddr",        KernelArg(accPtr));                                // 46  u64 (ptr) offset 216
    argMap.emplace("aOffset",        KernelArg(static_cast<std::uint64_t>(0)));         // 47  u64 offset 224

    auto args = buildArgsFromBinary(bin, argMap);
    if (args.empty())
    {
        std::cerr << "GPU: failed to build kernel arguments from m_argList\n";
        return {};
    }

    // -----------------------------------------------------------------------
    // DIAGNOSTICS (BEFORE LAUNCH)
    // -----------------------------------------------------------------------
    std::cerr << "\n[diag] === pre-launch state ===\n";

    // (1) Bias content
    std::cerr << "[diag] tensor.bias[0..3] (fp16->float): ";
    for (std::size_t i = 0; i < std::min<std::size_t>(4, tensor.bias.size()); ++i)
        std::cerr << float16ToFloat(tensor.bias[i]) << ' ';
    std::cerr << "\n[diag] biasFp32[0..3]: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(4, biasFp32.size()); ++i)
        std::cerr << biasFp32[i] << ' ';
    std::cerr << '\n';

    // (2) First few input/filter values
    std::cerr << "[diag] inputFp32  [0,1,100]: "
              << inputFp32[0] << ' ' << inputFp32[1] << ' '
              << (inputFp32.size() > 100 ? inputFp32[100] : 0.0f) << '\n';
    std::cerr << "[diag] filterFp32 [0,1,100]: "
              << filterFp32[0] << ' ' << filterFp32[1] << ' '
              << (filterFp32.size() > 100 ? filterFp32[100] : 0.0f) << '\n';

    // (3) Effective launch + arg dimensions
    std::cerr << "[diag] bin.m_grid   = {" << bin.m_grid.m_x   << ',' << bin.m_grid.m_y   << ',' << bin.m_grid.m_z   << "}\n";
    std::cerr << "[diag] bin.m_blocks = {" << bin.m_blocks.m_x << ',' << bin.m_blocks.m_y << ',' << bin.m_blocks.m_z << "}\n";
    std::cerr << "[diag] kNGroups     = " << kNGroups << '\n';

    auto dumpArg = [&](const char* name)
    {
        auto it = argMap.find(name);
        if (it == argMap.end()) { std::cerr << "[diag]   " << name << " : <missing>\n"; return; }
        std::cerr << "[diag]   " << name << " (size=" << it->second.size() << ") : ";
        const auto* p = static_cast<const std::uint8_t*>(it->second.data());
        for (std::size_t i = 0; i < it->second.size(); ++i)
            std::cerr << std::hex << static_cast<int>(p[i]) << ' ';
        std::cerr << std::dec << '\n';
    };
    for (const char* name : {
        "n","c","h","w","k","nGroups","flags64",
        "dataAddr","filterAddr","outputAddr","biasAddr",
        "alpha","beta",
        "dNStride","dCStride","dHStride",
        "fKStride","fCStride","fRStride",
        "oNStride","oKStride","oHStride",
        "activationMode","syncLimit","syncAddr","accAddr","aOffset"})
    {
        dumpArg(name);
    }

    // Use the binary's grid (producer-chosen, e.g. {96,1,1} for this shape).
    // Block dim is FIXED at 256 by the kernel ABI — do NOT use bin.m_blocks
    // (it's {1,1,1} for the winograd binaries and is not the HIP block dim).
    const dim3 grid (bin.m_grid.m_x, bin.m_grid.m_y, bin.m_grid.m_z);
    const dim3 block(256, 1, 1);
    if (!kernel.run(args, grid, block))
    {
        std::cerr << "GPU: kernel launch failed\n";
        return {};
    }

    if constexpr (std::is_same_v<Shader, HipShader>)
    {
        // Surface silent launch faults (e.g. OOB writes) that would otherwise
        // leave the output buffer at its zero-initialized state.
        hipError_t syncErr = hipDeviceSynchronize();
        std::cerr << "[diag] hipDeviceSynchronize: " << hipGetErrorString(syncErr)
                  << " (" << static_cast<int>(syncErr) << ")\n";
        if (syncErr != hipSuccess)
        {
            std::cerr << "GPU: hipDeviceSynchronize after launch failed\n";
            return {};
        }

        // (4) Peek at acc + output buffers right after sync
        std::vector<float> peekAcc(16, 0.0f);
        std::vector<float> peekOut(16, 0.0f);
        (void)hipMemcpy(peekAcc.data(), devAcc.data(),
                        sizeof(float) * peekAcc.size(), hipMemcpyDeviceToHost);
        (void)hipMemcpy(peekOut.data(), devOutput.data(),
                        sizeof(float) * peekOut.size(), hipMemcpyDeviceToHost);
        std::cerr << "[diag] devAcc   [0..15]: ";
        for (float v : peekAcc) std::cerr << v << ' ';
        std::cerr << "\n[diag] devOutput[0..15]: ";
        for (float v : peekOut) std::cerr << v << ' ';
        std::cerr << "\n[diag] === end ===\n\n";
    }

    std::vector<float> outFp32(tensor.output.size());
    devOutput.download(outFp32);

    return outFp32;
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

// ---------------------------------------------------------------------------
// HOST CONV COMPUTE
// ---------------------------------------------------------------------------
void computeConvCpu(
    const std::vector<float16>& input,
    const std::vector<float16>& filter,
    const std::vector<float16>& bias,
    TensorHost<float> &result,
    uint32_t N,          // batch size
    uint32_t Cg,          // input channels
    uint32_t H,          // input height
    uint32_t W,          // input width
    uint32_t Kg,          // output channels
    uint32_t R,
    uint32_t S,
    uint32_t G,
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
    const int C = G * Cg;
    const int K = G * Kg;

    std::vector<float> output(n * outH * outW * k, 0.0f);
    std::fill(output.begin(), output.end(), 0.0f);

    for (int n = 0; n < N; ++n)
    for (int g = 0; g < G; ++g)
    for (int k = 0; k < Kg; ++k)
    {
        const int k_global = g * Kg + k;
        for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow)
        {
            float acc = 0.0f;
            for (int c = 0; c < Cg; ++c)
            {
                const int c_global = g * Cg + c;
                for (int r = 0; r < R; ++r)
                for (int s = 0; s < S; ++s)
                {
                    const int ih = oh * convStrideX + r - startPadX;
                    const int iw = ow * convStrideY + s - startPadY;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    {
                        float d = input[n * C * H * W + c_global * H * W + ih * W + iw];
                        float f = filter[k_global * Cg * R * S + c * R * S + r * S + s];
                        acc += d * f;
                    }
                }
            }
            float val = acc * 1.0f + bias[k_global];  // fused bias (F_BIAS), then ReLU
            output[n * K * outH * outW + k_global * outH * outW + oh * outW + ow]
                = val < 0.0f ? 0.0f : val;  // ReLU (activation_mode=RELU=4)
        }
    }

    std::memcpy(result.data(), output.data(), output.size() * sizeof(float));
    
    // Flatten and print results one float per line
    std::vector<float> flatResults(result.data(), result.data() + (n * outH * outW * k));
    std::cout << "CPU Convolution Output Results (flattened):" << std::endl;
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

        MLSSstring asic = const_cast<MLSSstring>(MLSS_GFXAUTOFIND);
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
        // inputs --------------------------------------------------------------
        for (std::size_t i = 0; i < d.tensors.input.size(); ++i) d.tensors.input[i] = static_cast<float>(i % 64) / 64.0f; 
        // filter --------------------------------------------------------------
        for (std::size_t i = 0; i < d.tensors.filter.size(); ++i)
        {
            float v = static_cast<float>(i % 16) / 16.0f;
            d.tensors.filter[i] = ((i / (c * r * s)) % 2 == 0) ? -v : v;
        }
        // bias --------------------------------------------------------------
        // Alternating bias: odd channels get +10, even channels get -10
        // Combined with the alternating-sign filter this exercises both bias and ReLU
        for (std::size_t i = 0; i < d.tensors.bias.size(); ++i) d.tensors.bias[i] = (i % 2 == 0) ? 10.0f : -10.0f;
        // output ------------------------------------------------------------
        std::fill(d.tensors.output.begin(), d.tensors.output.end(), float16{});

        std::vector<std::uint32_t> shape = {n, outH, outW, k};
        d.hostRef = TensorHost<float>(shape);  // pre-allocate
        computeConvCpu(d.tensors.input, d.tensors.filter, d.tensors.bias, d.hostRef, 
                                      n, c, h, w, k, r, s, groups, hasBias, outH, outW, 
                                      startPadY, startPadX, endPadY, endPadX, 
                                      convStrideY, convStrideX);
            
        d.gpuOutShape = {n, outH, outW, k};

        return d;
    }();
    return data;
}
} // namespace

#if MLSS_D3D_ENABLED
TEST_CASE("CONV: D3D12 relocatable", "[conv][d3d]")
{
    ConvTestData& td = getTestData();

    REQUIRE(td.gpuAvailable);
    REQUIRE(td.reloc != nullptr);

    constexpr float kTolerance = 1e-3f;   // matches the working sample
    auto result = runConvGpu<D3D12ManagedModule, D3D12DeviceMemory, D3D12Shader>(
                  *td.reloc, td.tensors);
     REQUIRE_FALSE(result.empty());
     CHECK(compareBuffers(vectorToTensor(result, td.gpuOutShape),
                         td.hostRef, kTolerance, "D3D vs Host"));
}
#endif

#if MLSS_HIP_ENABLED
TEST_CASE("CONV: HIP non-relocatable", "[conv][hip]")
{
    auto& td = getTestData();
    REQUIRE(td.gpuAvailable);
    REQUIRE(td.nonReloc != nullptr);

    constexpr float kTolerance = 1e-3f;   // matches the working sample

    auto result = runConvGpu<HipManagedModule, HipDeviceMemory, HipShader>(
                  *td.nonReloc, td.tensors);
    REQUIRE_FALSE(result.empty());
    CHECK(compareBuffers(vectorToTensor(result, td.gpuOutShape),
                         td.hostRef, kTolerance, "HIP vs Host"));
}
#endif
