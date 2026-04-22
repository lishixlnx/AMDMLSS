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

#ifdef MLSS_D3D_ENABLED
#include "d3d/memory.hpp"
#include "d3d/module.hpp"
#include "d3d/shader.hpp"
#else
struct D3D12ManagedModule {};
struct D3D12DeviceMemory {};
struct D3D12Shader {};
#endif

#ifdef MLSS_CL_ENABLED
#include "cl/context.hpp"
#include "cl/memory.hpp"
#include "cl/module.hpp"
#include "cl/shader.hpp"
#else
struct ClContext {};
struct ClManagedModule {};
struct ClDeviceMemory {};
struct ClShader {};
#endif

namespace
{

constexpr std::uint32_t kBatchSize  = 1;
constexpr std::uint32_t kQHeadNum   = 4;
constexpr std::uint32_t kKVHeadNum  = 2;
constexpr std::uint32_t kQSeq       = 64;
constexpr std::uint32_t kKVSeq      = 64;
constexpr std::uint32_t kHeadDim    = 48;
const     float         kScale      = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
constexpr float         kTolerance  = 5e-2f;

constexpr std::size_t kQSize   = static_cast<std::size_t>(kBatchSize) * kQHeadNum  * kQSeq  * kHeadDim;
constexpr std::size_t kKSize   = static_cast<std::size_t>(kBatchSize) * kKVHeadNum * kKVSeq * kHeadDim;
constexpr std::size_t kVSize   = kKSize;
constexpr std::size_t kOutSize = kQSize;

// ---------------------------------------------------------------------------
// MLSS C API — obtain compiled GQA binaries
// ---------------------------------------------------------------------------

struct MlssBinaries
{
    MLSSbinary* data    = nullptr;
    MLSSsize    count   = 0;
    MLSScontext context = 0;
};

MlssBinaries getGqaBinaries(MLSSstring asic)
{
    MlssBinaries out{};
    MLSSstring   opName   = const_cast<MLSSstring>(MLSS_GQA);
    MLSSenum     dataType = MLSS_FLOAT16;
    std::uint32_t packing  = MLSS_ATTR_CONFIG_GQA_PACKING_UNPACKED;
    std::uint32_t kvDim    = 0;
    std::uint32_t batch    = kBatchSize;
    std::uint32_t qHeads   = kQHeadNum;
    std::uint32_t kvHeads  = kKVHeadNum;
    std::uint32_t qSeq     = kQSeq;
    std::uint32_t kvSeq    = kKVSeq;
    std::uint32_t hDim     = kHeadDim;
    float         scale    = kScale;

    CHECK_STATUS(mlssCreateContext(&out.context, asic, opName));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_BATCH,       &batch));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_QSEQ,        &qSeq));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_KVSEQ,       &kvSeq));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_KDIM,        &kvDim));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_VDIM,        &kvDim));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_SIZEHEADS,   &hDim));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_PACKING,     &packing));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_QHEADCOUNT,  &qHeads));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_KVHEADCOUNT, &kvHeads));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_SCALE,       &scale));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_GQA_DATATYPE,    &dataType));

    MLSSstatus* pStatuses  = nullptr;
    MLSSsize    nStatuses  = 0;
    if (mlssGetCaps(out.context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
    {
        std::cerr << "GQA getCaps failed — configuration not supported on this ASIC\n";
        std::exit(kSkipExitCode);
    }

    CHECK_STATUS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;
}

// ---------------------------------------------------------------------------
// Host reference — grouped-query attention in fp32
// ---------------------------------------------------------------------------

TensorHost<float> runGqaHost(const TensorHost<float>& Q,
                             const TensorHost<float>& K,
                             const TensorHost<float>& V)
{
    return referenceAttention(Q, K, V, kScale);
}

// ---------------------------------------------------------------------------
// Templated GPU dispatch for HIP / D3D12 / OpenCL
// ---------------------------------------------------------------------------

template<typename Module, typename Memory, typename Shader>
std::vector<float> runGqaGpu(const MLSSbinary& bin,
                             const std::vector<float16_t>& hostQ,
                             const std::vector<float16_t>& hostK,
                             const std::vector<float16_t>& hostV)
{
    auto desc = buildShaderDescriptor(bin);

    // --- module construction -----------------------------------------------
    [[maybe_unused]] ClContext backendCtx{};
    Module module = [&]() -> Module
    {
        if constexpr (std::is_same_v<Module, ClManagedModule>)
            return Module(backendCtx, desc);
        else
            return Module(desc);
    }();

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
        if constexpr (std::is_same_v<Memory, D3D12DeviceMemory>)
            return Memory(module.getDevice()->handle().Get(), bytes);
        else if constexpr (std::is_same_v<Memory, ClDeviceMemory>)
            return Memory(backendCtx, bytes);
        else
            return Memory(bytes);
    };

    Memory devQ   = allocMem(fp16Bytes * kQSize);
    Memory devK   = allocMem(fp16Bytes * kKSize);
    Memory devV   = allocMem(fp16Bytes * kVSize);
    Memory devOut = allocMem(fp16Bytes * kOutSize);

    devQ.upload(hostQ);
    devK.upload(hostK);
    devV.upload(hostV);

    devQ.setDoublePointer();
    devK.setDoublePointer();
    devV.setDoublePointer();
    devOut.setDoublePointer();

    // --- kernel arguments (manual construction matching kernel ABI) ---------
    std::int32_t argBatch    = static_cast<std::int32_t>(kBatchSize);
    std::int32_t argQSeq     = static_cast<std::int32_t>(kQSeq);
    std::int32_t argKVSeq    = static_cast<std::int32_t>(kKVSeq);
    std::int32_t argQHeads   = static_cast<std::int32_t>(kQHeadNum);
    std::int32_t argKVHeads  = static_cast<std::int32_t>(kKVHeadNum);
    std::int32_t argHeadDim  = static_cast<std::int32_t>(kHeadDim);
    std::int32_t argDv       = static_cast<std::int32_t>(kHeadDim);
    float        argScale    = kScale;

    const auto [qStrides, kStrides, vStrides, outStrides] =
        calcStrides<GQAPackingFlags::UNPACKED_QUERY_ROW>(
            static_cast<index_t>(kBatchSize),
            static_cast<index_t>(kQHeadNum),
            static_cast<index_t>(kKVHeadNum),
            static_cast<index_t>(kQSeq),
            static_cast<index_t>(kKVSeq),
            static_cast<index_t>(kHeadDim));

    // CK kernel ABI order: Q**, K**, V**, Out**, M, N, K, O, G0, G1, G1kv, alpha, strides...
    // M=q_seq, N=kv_seq, K=head_dim, O=d_v, G0=batch, G1=q_heads, G1kv=kv_heads
    std::vector<KernelArg> args;
    args.reserve(28);

    args.push_back(KernelArg(devQ.getDoublePointer()));
    args.push_back(KernelArg(devK.getDoublePointer()));
    args.push_back(KernelArg(devV.getDoublePointer()));
    args.push_back(KernelArg(devOut.getDoublePointer()));

    args.push_back(KernelArg(argQSeq));
    args.push_back(KernelArg(argKVSeq));
    args.push_back(KernelArg(argHeadDim));
    args.push_back(KernelArg(argDv));
    args.push_back(KernelArg(argBatch));
    args.push_back(KernelArg(argQHeads));
    args.push_back(KernelArg(argKVHeads));
    args.push_back(KernelArg(argScale));

    for (std::uint32_t d = 0; d < 4; ++d) args.push_back(KernelArg(qStrides[d]));
    for (std::uint32_t d = 0; d < 4; ++d) args.push_back(KernelArg(kStrides[d]));
    for (std::uint32_t d = 0; d < 4; ++d) args.push_back(KernelArg(vStrides[d]));
    for (std::uint32_t d = 0; d < 4; ++d) args.push_back(KernelArg(outStrides[d]));

    // --- launch ------------------------------------------------------------
    const dim3 grid(bin.m_grid.m_x,   bin.m_grid.m_y,   bin.m_grid.m_z);
    const dim3 block(bin.m_blocks.m_x, bin.m_blocks.m_y, bin.m_blocks.m_z);

    if (!kernel.run(args, grid, block))
    {
        std::cerr << "GPU: kernel launch failed\n";
        return {};
    }

    // --- readback ----------------------------------------------------------
    std::vector<float16_t> outFp16(kOutSize);
    devOut.download(outFp16);
    return halvesToFloats(outFp16);
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

        MLSSstring asic = const_cast<MLSSstring>(MLSS_GFXAUTOFIND);
        CHECK_STATUS(mlssSetVerboseLevel(0));

        auto mlssBins = getGqaBinaries(asic);
        std::cout << "GQA binaries obtained: " << mlssBins.count << " blobs\n";

        const MLSSbinary* nonReloc = findBinary(mlssBins.data, mlssBins.count, false);
        const MLSSbinary* reloc    = findBinary(mlssBins.data, mlssBins.count, true);

        if (nonReloc == nullptr && reloc == nullptr)
        {
            std::cerr << "FAIL: no usable binary found\n";
            return EXIT_FAILURE;
        }

        auto hostQ = generateRandomTensor({kBatchSize, kQHeadNum, kQSeq, kHeadDim},  -0.5f, 0.5f, 10);
        auto hostK = generateRandomTensor({kBatchSize, kKVHeadNum, kKVSeq, kHeadDim}, -0.5f, 0.5f, 20);
        auto hostV = generateRandomTensor({kBatchSize, kKVHeadNum, kKVSeq, kHeadDim}, -0.5f, 0.5f, 30);

        auto gpuQ  = transposeHeadSeq(hostQ);
        auto gpuK  = transposeHeadSeq(hostK);
        auto gpuV  = transposeHeadSeq(hostV);
        auto hostQ16 = tensorToHalves(gpuQ);
        auto hostK16 = tensorToHalves(gpuK);
        auto hostV16 = tensorToHalves(gpuV);

        auto hostRef = runGqaHost(hostQ, hostK, hostV);

        bool allPassed = true;

        const std::vector<std::uint32_t> gpuOutShape = {kBatchSize, kQSeq, kQHeadNum, kHeadDim};

        // --- HIP (non-relocatable) -----------------------------------------
        if (nonReloc != nullptr)
        {
            std::cout << "\n--- HIP (non-relocatable) ---\n";
            auto resultVec = runGqaGpu<HipManagedModule, HipDeviceMemory, HipShader>(
                                 *nonReloc, hostQ16, hostK16, hostV16);
            if (resultVec.empty())
            {
                std::cerr << "FAIL: HIP GQA kernel produced no results\n";
                allPassed = false;
            }
            else if (!compareBuffers(transposeHeadSeq(vectorToTensor(resultVec, gpuOutShape)),
                                     hostRef, kTolerance, "HIP vs Host"))
            {
                allPassed = false;
            }
            else
            {
                std::cout << "PASS: HIP vs Host (tolerance=" << kTolerance << ")\n";
            }
        }

        // --- D3D12 (relocatable) -------------------------------------------
#ifdef MLSS_D3D_ENABLED
        if (reloc != nullptr)
        {
            std::cout << "\n--- D3D12 (relocatable) ---\n";
            auto resultVec = runGqaGpu<D3D12ManagedModule, D3D12DeviceMemory, D3D12Shader>(
                                 *reloc, hostQ16, hostK16, hostV16);
            if (resultVec.empty())
            {
                std::cerr << "FAIL: D3D GQA kernel produced no results\n";
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
            auto resultVec = runGqaGpu<ClManagedModule, ClDeviceMemory, ClShader>(
                                 *nonReloc, hostQ16, hostK16, hostV16);
            if (resultVec.empty())
            {
                std::cerr << "WARN: CL GQA kernel produced no results\n";
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

        std::cout << "\n=== GQA Test " << (allPassed ? "PASSED" : "FAILED") << " ===\n";
        return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "FATAL: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "FATAL: unknown exception\n";
        return EXIT_FAILURE;
    }
}
