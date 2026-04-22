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
#endif

#ifdef MLSS_CL_ENABLED
#include "cl/context.hpp"
#include "cl/memory.hpp"
#include "cl/module.hpp"
#include "cl/shader.hpp"
#endif

namespace
{

constexpr std::uint32_t kBatchSize = 1;
constexpr std::uint32_t kHeadNum   = 2;
constexpr std::uint32_t kQSeq      = 64;
constexpr std::uint32_t kKVSeq     = 32;
constexpr std::uint32_t kHeadDim   = 48;
const     float         kScale     = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
constexpr float         kTolerance = 5e-2f;

constexpr std::size_t kQSize   = static_cast<std::size_t>(kBatchSize) * kHeadNum * kQSeq  * kHeadDim;
constexpr std::size_t kKSize   = static_cast<std::size_t>(kBatchSize) * kHeadNum * kKVSeq * kHeadDim;
constexpr std::size_t kVSize   = kKSize;
constexpr std::size_t kOutSize = kQSize;

// ---------------------------------------------------------------------------
// MLSS C API — obtain compiled MHA binaries
// ---------------------------------------------------------------------------

struct MlssBinaries
{
    MLSSbinary* data    = nullptr;
    MLSSsize    count   = 0;
    MLSScontext context = 0;
};

MlssBinaries getMhaBinaries(MLSSstring asic)
{
    MlssBinaries out{};
    MLSSstring   opName   = const_cast<MLSSstring>(MLSS_MHA);
    MLSSenum     dataType = MLSS_FLOAT16;
    std::uint32_t packing = MLSS_ATTR_CONFIG_MHA_PACKING_UNPACKED;
    std::uint32_t kvDim   = 0;
    std::uint32_t batch   = kBatchSize;
    std::uint32_t heads   = kHeadNum;
    std::uint32_t qSeq    = kQSeq;
    std::uint32_t kvSeq   = kKVSeq;
    std::uint32_t hDim    = kHeadDim;
    float         scale   = kScale;

    CHECK_STATUS(mlssCreateContext(&out.context, asic, opName));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_BATCH,     &batch));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_QSEQ,      &qSeq));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_KVSEQ,     &kvSeq));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_KDIM,      &kvDim));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_VDIM,      &kvDim));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_SIZEHEADS, &hDim));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_PACKING,   &packing));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_HEADCOUNT, &heads));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_SCALE,     &scale));
    CHECK_STATUS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_MHA_DATATYPE,  &dataType));

    MLSSstatus* pStatuses  = nullptr;
    MLSSsize    nStatuses  = 0;
    if (mlssGetCaps(out.context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
    {
        std::cerr << "MHA getCaps failed — configuration not supported on this ASIC\n";
        std::exit(kSkipExitCode);
    }

    CHECK_STATUS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;
}

// ---------------------------------------------------------------------------
// Host reference — scaled dot-product attention in fp32
// ---------------------------------------------------------------------------

TensorHost<float> runMhaHost(const TensorHost<float>& Q,
                             const TensorHost<float>& K,
                             const TensorHost<float>& V)
{
    return referenceAttention(Q, K, V, kScale);
}

// ---------------------------------------------------------------------------
// Templated GPU dispatch for HIP / D3D12 / OpenCL
// ---------------------------------------------------------------------------

template<typename Module, typename Memory, typename Shader>
std::vector<float> runMhaGpu(const MLSSbinary& bin,
                             const std::vector<float16_t>& hostQ,
                             const std::vector<float16_t>& hostK,
                             const std::vector<float16_t>& hostV)
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
                if (!names.empty())
                    sh = module.getShader(names[0]);
            }
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

    // --- kernel arguments (built dynamically from m_argList) ----------------
    std::int32_t argBatch   = static_cast<std::int32_t>(kBatchSize);
    std::int32_t argQSeq    = static_cast<std::int32_t>(kQSeq);
    std::int32_t argKVSeq   = static_cast<std::int32_t>(kKVSeq);
    std::int32_t argHeadNum = static_cast<std::int32_t>(kHeadNum);
    std::int32_t argHeadDim = static_cast<std::int32_t>(kHeadDim);
    float        argScale   = kScale;

    std::unordered_map<std::string, KernelArg> argMap;
    argMap.emplace("Q",      KernelArg(devQ.getDoublePointer()));
    argMap.emplace("K",      KernelArg(devK.getDoublePointer()));
    argMap.emplace("V",      KernelArg(devV.getDoublePointer()));
    argMap.emplace("output", KernelArg(devOut.getDoublePointer()));

    argMap.emplace("batch_size",         KernelArg(argBatch));
    argMap.emplace("q_sequence_length",  KernelArg(argQSeq));
    argMap.emplace("kv_sequence_length", KernelArg(argKVSeq));
    argMap.emplace("head_num",           KernelArg(argHeadNum));
    argMap.emplace("head_dim",           KernelArg(argHeadDim));
    argMap.emplace("scale",              KernelArg(argScale));

    const auto [qStrides, kStrides, vStrides, outStrides] =
        calcStrides<GQAPackingFlags::UNPACKED_QUERY_ROW>(
            static_cast<index_t>(kBatchSize),
            static_cast<index_t>(kHeadNum),
            static_cast<index_t>(kHeadNum),
            static_cast<index_t>(kQSeq),
            static_cast<index_t>(kKVSeq),
            static_cast<index_t>(kHeadDim));

    for (std::uint32_t d = 0; d < 4; ++d)
    {
        argMap.emplace("q_stride_d"      + std::to_string(d), KernelArg(qStrides[d]));
        argMap.emplace("k_stride_d"      + std::to_string(d), KernelArg(kStrides[d]));
        argMap.emplace("v_stride_d"      + std::to_string(d), KernelArg(vStrides[d]));
        argMap.emplace("output_stride_d" + std::to_string(d), KernelArg(outStrides[d]));
    }

    auto args = buildArgsFromBinary(bin, argMap);
    if (args.empty())
    {
        std::cerr << "GPU: failed to build kernel arguments from m_argList\n";
        return {};
    }

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

        auto mlssBins = getMhaBinaries(asic);
        std::cout << "MHA binaries obtained: " << mlssBins.count << " blobs\n";

        const MLSSbinary* nonReloc = findBinary(mlssBins.data, mlssBins.count, false);
        const MLSSbinary* reloc    = findBinary(mlssBins.data, mlssBins.count, true);

        if (nonReloc == nullptr && reloc == nullptr)
        {
            std::cerr << "FAIL: no usable binary found\n";
            return EXIT_FAILURE;
        }

        auto hostQ = generateRandomTensor({kBatchSize, kHeadNum, kQSeq, kHeadDim},  -0.5f, 0.5f, 1);
        auto hostK = generateRandomTensor({kBatchSize, kHeadNum, kKVSeq, kHeadDim}, -0.5f, 0.5f, 2);
        auto hostV = generateRandomTensor({kBatchSize, kHeadNum, kKVSeq, kHeadDim}, -0.5f, 0.5f, 3);

        auto gpuQ  = transposeHeadSeq(hostQ);
        auto gpuK  = transposeHeadSeq(hostK);
        auto gpuV  = transposeHeadSeq(hostV);
        auto hostQ16 = tensorToHalves(gpuQ);
        auto hostK16 = tensorToHalves(gpuK);
        auto hostV16 = tensorToHalves(gpuV);

        auto hostRef = runMhaHost(hostQ, hostK, hostV);

        bool allPassed = true;

        const std::vector<std::uint32_t> gpuOutShape = {kBatchSize, kQSeq, kHeadNum, kHeadDim};

        // --- HIP (non-relocatable) -----------------------------------------
        if (nonReloc != nullptr)
        {
            std::cout << "\n--- HIP (non-relocatable) ---\n";
            auto resultVec = runMhaGpu<HipManagedModule, HipDeviceMemory, HipShader>(
                                 *nonReloc, hostQ16, hostK16, hostV16);
            if (resultVec.empty())
            {
                std::cerr << "FAIL: HIP MHA kernel produced no results\n";
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
    catch (...)
    {
        std::cerr << "FATAL: unknown exception\n";
        return EXIT_FAILURE;
    }
}
