/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * HIP validation for Conv MxN Winograd Base — MIGraphX consumer pattern.
 *
 * AMDMLSS is used read-only:
 *   1. mlssCreateContext(MLSS_GFX1100) to fetch gfx1100 archived bins.
 *   2. Select non-reloc ELF, kernel "main".
 *   3. Consumer-side ELF metadata patch (gfx1100 -> runtime gfx) before load.
 *   4. hipModuleLoadData + hipModuleGetFunction("main") + launch.
 *   5. Compare GPU output against a CPU reference.
 *
 * No AMDMLSS library changes are required for this flow.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include <amdmlss/amdmlss_api.h>

namespace
{
constexpr std::uint32_t kBlockDimX = 256U;
constexpr std::uint8_t kKernelActivationRelu = 4U;

constexpr MLSSuint32 w = 112U;
constexpr MLSSuint32 h = 112U;
constexpr MLSSuint32 c = 64U;
constexpr MLSSuint32 n = 1U;
constexpr MLSSuint32 k = 128U;
constexpr MLSSuint32 s = 3U;
constexpr MLSSuint32 r = 3U;
constexpr MLSSuint32 startPadX = 1U;
constexpr MLSSuint32 startPadY = 1U;
constexpr MLSSuint32 endPadX = 1U;
constexpr MLSSuint32 endPadY = 1U;
constexpr MLSSuint32 outPadX = 0U;
constexpr MLSSuint32 outPadY = 0U;
constexpr MLSSuint32 convStrideX = 1U;
constexpr MLSSuint32 convStrideY = 1U;
constexpr MLSSuint32 outW = 112U;
constexpr MLSSuint32 outH = 112U;
constexpr MLSSuint32 inputStrideX = 1U;
constexpr MLSSuint32 inputStrideY = 1U;
constexpr MLSSuint32 filterStrideX = 1U;
constexpr MLSSuint32 filterStrideY = 1U;
constexpr MLSSuint32 groups = 1U;
constexpr MLSSbool hasBias = true;
constexpr MLSSbool crossCorrelation = false;
constexpr MLSSbool backward = false;

constexpr MLSSuint32 dNStride = groups * (c / groups) * h * w;
constexpr MLSSuint32 dHStride = w;
constexpr MLSSuint32 dCStride = h * w;
constexpr MLSSuint32 fKStride = (c / groups) * r * s;
constexpr MLSSuint32 fCStride = r * s;
constexpr MLSSuint32 fRStride = s;
constexpr MLSSuint32 fSStride = 1U;
constexpr MLSSuint32 oNStride = groups * (k / groups) * outH * outW;
constexpr MLSSuint32 oHStride = outW;
constexpr MLSSuint32 oKStride = outH * outW;
constexpr MLSSuint32 dOffset = 0U;
constexpr MLSSuint32 oOffset = 0U;
constexpr MLSSuint32 fOffset = 0U;
constexpr MLSSuint32 bOffset = 0U;

constexpr MLSSuint32 activation = MLSS_ACTIVATION_RELU;
constexpr MLSSuint32 precision = MLSS_PRECISION_FLOAT16_ADD_FLOAT32;
constexpr MLSSenum dataType = MLSS_FLOAT32;

constexpr float kCompareTolerance = 1.0e-3F;
constexpr std::size_t kMaxPrintedMismatches = 8U;

// Winograd Base non-reloc archives ship as gfx1100 ELFs (see AMDMLSS shadersUtils).
constexpr const char* kArchiveGfx = "gfx1100";
constexpr std::size_t kElf64EflagsOffset = 0x30U;
constexpr std::uint8_t kEfAmdgpuMachMask = 0xFFU;
constexpr std::uint8_t kMachGfx1100 = 0x041U;

struct GfxMachEntry
{
    const char* gfxName;
    std::uint8_t machCode;
};

constexpr GfxMachEntry kGfxMachTable[] = {
    {"gfx1100", 0x041U},
    {"gfx1101", 0x046U},
    {"gfx1102", 0x047U},
    {"gfx1150", 0x043U},
    {"gfx1151", 0x04AU},
    {"gfx1152", 0x055U},
    {"gfx1153", 0x058U},
};

std::optional<std::uint8_t> machCodeForGfx(std::string_view runtimeGfx)
{
    for (const GfxMachEntry& entry : kGfxMachTable)
    {
        if (runtimeGfx == entry.gfxName)
            return entry.machCode;
    }
    return std::nullopt;
}

std::vector<std::uint8_t> patchGfxMetadata(
    std::span<const std::uint8_t> input,
    std::string_view fromGfx,
    std::uint8_t machFrom,
    std::string_view toGfx,
    std::uint8_t machTo)
{
    if (fromGfx.size() != toGfx.size())
        return {};

    std::vector<std::uint8_t> patched(input.begin(), input.end());

    if (patched.size() > kElf64EflagsOffset
        && (patched[kElf64EflagsOffset] & kEfAmdgpuMachMask) == machFrom)
    {
        patched[kElf64EflagsOffset] =
            static_cast<std::uint8_t>((patched[kElf64EflagsOffset] & ~kEfAmdgpuMachMask) | machTo);
    }

    for (std::size_t i = 0; i + fromGfx.size() <= patched.size(); ++i)
    {
        if (std::memcmp(patched.data() + i, fromGfx.data(), fromGfx.size()) == 0)
            std::memcpy(patched.data() + i, toGfx.data(), toGfx.size());
    }

    return patched;
}

// MIGraphX-style: copy AMDMLSS bin and patch ELF target to the runtime GPU when needed.
std::vector<std::uint8_t> prepareModuleImage(const MLSSbinary& bin, std::string_view runtimeGfx)
{
    const auto* raw = static_cast<const std::uint8_t*>(bin.m_binaries);
    std::vector<std::uint8_t> image(raw, raw + bin.m_binarySize);

    if (runtimeGfx == kArchiveGfx)
    {
        std::cout << "Module image:  no metadata patch (runtime == " << kArchiveGfx << ")\n";
        return image;
    }

    const auto machTo = machCodeForGfx(runtimeGfx);
    if (!machTo.has_value())
    {
        std::cerr << "No mach mapping for runtime gfx '" << runtimeGfx
                  << "'; cannot patch archived " << kArchiveGfx << " ELF\n";
        std::exit(EXIT_FAILURE);
    }

    image = patchGfxMetadata(image, kArchiveGfx, kMachGfx1100, runtimeGfx, *machTo);
    if (image.empty())
    {
        std::cerr << "ELF metadata patch failed\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Module image:  patched ELF metadata " << kArchiveGfx << " -> " << runtimeGfx << '\n';
    return image;
}

void fillSimpleTestData(
    std::vector<float>& hostInput,
    std::vector<float>& hostFilter,
    std::vector<float>& hostBias)
{
    // Input [N,C,H,W]: channel 0 = 1.0, channel 1 = 2.0, ...
    for (std::uint32_t ci = 0; ci < c; ++ci)
    {
        for (std::uint32_t yi = 0; yi < h; ++yi)
        {
            for (std::uint32_t xi = 0; xi < w; ++xi)
            {
                const std::size_t idx =
                    static_cast<std::size_t>(ci) * h * w + static_cast<std::size_t>(yi) * w + xi;
                hostInput[idx] = static_cast<float>(ci + 1U);
            }
        }
    }

    // Filter [K,C/g,R,S]: output channel ko uses only input channel (ko % C) with a 3x3 ones patch.
    std::fill(hostFilter.begin(), hostFilter.end(), 0.0F);
    const std::uint32_t channelsPerGroup = c / groups;
    for (std::uint32_t ko = 0; ko < k; ++ko)
    {
        const std::uint32_t ci = ko % channelsPerGroup;
        for (std::uint32_t ry = 0; ry < r; ++ry)
        {
            for (std::uint32_t rx = 0; rx < s; ++rx)
            {
                const std::size_t idx =
                    static_cast<std::size_t>(ko) * channelsPerGroup * r * s
                    + static_cast<std::size_t>(ci) * r * s
                    + static_cast<std::size_t>(ry) * s
                    + rx;
                hostFilter[idx] = 1.0F;
            }
        }
    }

    // Bias [K]: small alternating values to exercise fused bias + ReLU.
    for (std::uint32_t ko = 0; ko < k; ++ko)
        hostBias[ko] = (ko % 2U == 0U) ? 0.5F : -0.5F;
}

void computeConvCpuRef(
    const std::vector<float>& input,
    const std::vector<float>& filter,
    const std::vector<float>& bias,
    std::vector<float>& output)
{
    const std::uint32_t channelsPerGroup = c / groups;
    const std::uint32_t totalChannels = groups * channelsPerGroup;
    const std::uint32_t totalFeatures = groups * k;

    std::fill(output.begin(), output.end(), 0.0F);

    for (std::uint32_t ni = 0; ni < n; ++ni)
    {
        for (std::uint32_t gi = 0; gi < groups; ++gi)
        {
            for (std::uint32_t ko = 0; ko < k; ++ko)
            {
                const std::uint32_t kGlobal = gi * k + ko;
                for (std::uint32_t oh = 0; oh < outH; ++oh)
                {
                    for (std::uint32_t ow = 0; ow < outW; ++ow)
                    {
                        float acc = 0.0F;
                        for (std::uint32_t ci = 0; ci < channelsPerGroup; ++ci)
                        {
                            const std::uint32_t cGlobal = gi * channelsPerGroup + ci;
                            for (std::uint32_t ry = 0; ry < r; ++ry)
                            {
                                for (std::uint32_t rx = 0; rx < s; ++rx)
                                {
                                    const int ih = static_cast<int>(oh * convStrideY + ry) - static_cast<int>(startPadY);
                                    const int iw = static_cast<int>(ow * convStrideX + rx) - static_cast<int>(startPadX);
                                    if (ih >= 0 && ih < static_cast<int>(h)
                                        && iw >= 0 && iw < static_cast<int>(w))
                                    {
                                        const std::size_t inIdx =
                                            static_cast<std::size_t>(ni) * totalChannels * h * w
                                            + static_cast<std::size_t>(cGlobal) * h * w
                                            + static_cast<std::size_t>(ih) * w
                                            + static_cast<std::size_t>(iw);
                                        const std::size_t filtIdx =
                                            static_cast<std::size_t>(kGlobal) * channelsPerGroup * r * s
                                            + static_cast<std::size_t>(ci) * r * s
                                            + static_cast<std::size_t>(ry) * s
                                            + rx;
                                        acc += input[inIdx] * filter[filtIdx];
                                    }
                                }
                            }
                        }

                        float value = acc + (hasBias ? bias[kGlobal] : 0.0F);
                        if (value < 0.0F)
                            value = 0.0F;

                        const std::size_t outIdx =
                            static_cast<std::size_t>(ni) * totalFeatures * outH * outW
                            + static_cast<std::size_t>(kGlobal) * outH * outW
                            + static_cast<std::size_t>(oh) * outW
                            + ow;
                        output[outIdx] = value;
                    }
                }
            }
        }
    }
}

bool compareOutputs(
    const std::vector<float>& gpu,
    const std::vector<float>& cpu,
    float tolerance)
{
    if (gpu.size() != cpu.size())
    {
        std::cerr << "Compare FAIL: size mismatch " << gpu.size() << " vs " << cpu.size() << '\n';
        return false;
    }

    float maxDiff = 0.0F;
    std::size_t mismatchCount = 0U;

    for (std::size_t i = 0; i < gpu.size(); ++i)
    {
        const float diff = std::fabs(gpu[i] - cpu[i]);
        maxDiff = std::max(maxDiff, diff);
        if (diff > tolerance)
        {
            if (mismatchCount < kMaxPrintedMismatches)
            {
                std::cerr << "  mismatch[" << i << "]: gpu=" << gpu[i]
                          << " cpu=" << cpu[i] << " diff=" << diff << '\n';
            }
            ++mismatchCount;
        }
    }

    std::cout << "CPU vs GPU:    maxDiff=" << maxDiff
              << " tolerance=" << tolerance
              << " mismatches=" << mismatchCount << " / " << gpu.size() << '\n';

    if (mismatchCount > kMaxPrintedMismatches)
        std::cerr << "  ... and " << (mismatchCount - kMaxPrintedMismatches) << " more mismatches\n";

    return mismatchCount == 0U;
}

void checkStatus(MLSSstatus status, int line)
{
    if (status != MLSS_SUCCESS)
    {
        MLSSstring err = mlssGetErrorString(status);
        std::cerr << "MLSS FAIL at line " << line << ": " << err << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_MLSS(status) checkStatus((status), __LINE__)

void checkHip(hipError_t status, const char* where)
{
    if (status != hipSuccess)
    {
        std::cerr << "HIP FAIL at " << where << ": " << hipGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP(expr) checkHip((expr), #expr)

class KernelArg
{
public:
    KernelArg() = default;

    template<typename T>
    explicit KernelArg(T* ptr)
    {
        std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
        add(&addr, sizeof(std::uintptr_t));
    }

    template<typename T>
    explicit KernelArg(const T& value)
    {
        add(&value, sizeof(T));
    }

    void add(const void* ptr, std::size_t size)
    {
        m_buffer.resize(size);
        std::memcpy(m_buffer.data(), ptr, size);
    }

    const void* data() const { return m_buffer.data(); }
    std::size_t size() const { return m_buffer.size(); }

    static void serialize(const std::vector<KernelArg>& args, std::vector<std::byte>& out)
    {
        if (args.empty())
        {
            out.clear();
            return;
        }

        std::size_t totalSize = args[0].size();
        for (std::size_t idx = 1; idx < args.size(); ++idx)
        {
            const std::size_t alignment = args[idx].size();
            const std::size_t padding = (alignment - (totalSize % alignment)) % alignment;
            totalSize += padding + args[idx].size();
        }

        out.resize(totalSize);
        std::size_t offset = args[0].size();
        std::memcpy(out.data(), args[0].data(), args[0].size());

        for (std::size_t idx = 1; idx < args.size(); ++idx)
        {
            const std::size_t alignment = args[idx].size();
            const std::size_t padding = (alignment - (offset % alignment)) % alignment;
            const std::size_t writeOffset = offset + padding;
            std::memcpy(out.data() + writeOffset, args[idx].data(), args[idx].size());
            offset = writeOffset + args[idx].size();
        }
    }

private:
    std::vector<std::byte> m_buffer;
};

const MLSSbinary* findNonRelocMainKernel(const MLSSbinary* binaries, MLSSsize count)
{
    const MLSSbinary* fallback = nullptr;

    for (const auto* p = binaries; p < binaries + count; ++p)
    {
        if (p->m_isRelocatable)
            continue;

        if (p->m_pKernelName != nullptr && std::strcmp(p->m_pKernelName, "main") == 0)
            return p;

        if (fallback == nullptr || p->m_argList.m_size > fallback->m_argList.m_size)
            fallback = p;
    }

    return fallback;
}

std::vector<KernelArg> buildArgsFromBinary(
    const MLSSbinary& bin,
    const std::unordered_map<std::string, KernelArg>& argMap)
{
    MLSSvoid* rawData = nullptr;
    MLSSsize argCount = 0;
    MLSSenum argType = 0;

    if (mlssVectorRetrieveData(bin.m_argList, &rawData, &argCount, &argType) != MLSS_SUCCESS
        || rawData == nullptr || argCount == 0)
    {
        std::cerr << "buildArgsFromBinary: failed to retrieve m_argList\n";
        return {};
    }

    const auto* mlssArgs = static_cast<const MLSSarg*>(rawData);

    std::vector<std::size_t> order(static_cast<std::size_t>(argCount));
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
    {
        return mlssArgs[a].m_place < mlssArgs[b].m_place;
    });

    std::vector<KernelArg> args;
    args.reserve(static_cast<std::size_t>(argCount));

    for (std::size_t idx : order)
    {
        const MLSSarg& arg = mlssArgs[idx];
        const std::string name = arg.m_name ? arg.m_name : "";
        const auto it = argMap.find(name);
        if (it == argMap.end())
        {
            std::cerr << "buildArgsFromBinary: missing argument '" << name << "'\n";
            return {};
        }
        args.push_back(it->second);
    }

    return args;
}

struct MlssBinaries
{
    MLSSbinary* data = nullptr;
    MLSSsize count = 0;
    MLSScontext context = 0;
};

MlssBinaries queryConvBinaries(MLSSstring asic)
{
    MlssBinaries out{};
    MLSSstring opName = const_cast<MLSSstring>(MLSS_CONV);

    CHECK_MLSS(mlssCreateContext(&out.context, asic, opName));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_W, &w));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_H, &h));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_C, &c));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_N, &n));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_K, &k));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_S, &s));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_R, &r));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTW, &outW));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTH, &outH));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_STARTPADX, &startPadX));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_STARTPADY, &startPadY));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ENDPADX, &endPadX));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ENDPADY, &endPadY));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTPADX, &outPadX));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OUTPADY, &outPadY));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_CONVSTRIDEX, &convStrideX));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_CONVSTRIDEY, &convStrideY));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_INPUTSTRIDEX, &inputStrideX));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_INPUTSTRIDEY, &inputStrideY));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FILTERSTRIDEX, &filterStrideX));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FILTERSTRIDEY, &filterStrideY));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_GROUPS, &groups));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_HASBIAS, &hasBias));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_CROSSCORRELATION, &crossCorrelation));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_BACKWARD, &backward));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DNSTRIDE, &dNStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DHSTRIDE, &dHStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DCSTRIDE, &dCStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FKSTRIDE, &fKStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FCSTRIDE, &fCStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FRSTRIDE, &fRStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FSSTRIDE, &fSStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ONSTRIDE, &oNStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OHSTRIDE, &oHStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OKSTRIDE, &oKStride));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DOFFSET, &dOffset));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_OOFFSET, &oOffset));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_FOFFSET, &fOffset));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_BOFFSET, &bOffset));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_DATATYPE, &dataType));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_PRECISION, &precision));
    CHECK_MLSS(mlssSetParameterByEnum(&out.context, opName, MLSS_ATTR_CONV_ACTIVATION, &activation));

    MLSSstatus* pStatuses = nullptr;
    MLSSsize nStatuses = 0;
    if (mlssGetCaps(out.context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
    {
        std::cerr << "mlssGetCaps failed for ASIC " << asic << '\n';
        std::exit(EXIT_FAILURE);
    }

    CHECK_MLSS(mlssGetBinaries(out.context, &out.data, &out.count));
    return out;
}

} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    mlssSetVerboseLevel(2);

    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    std::cout << "Runtime GPU: " << props.gcnArchName << " (" << props.name << ")\n";

    // MIGraphX-style: query gfx1100 bins even when running on gfx1151 hardware.
    const MLSSstring queryAsic = const_cast<MLSSstring>(MLSS_GFX1100);
    std::cout << "Query ASIC:    " << queryAsic << " (archived bin; runtime patch in sample)\n";

    MlssBinaries bins = queryConvBinaries(queryAsic);
    std::cout << "Binaries:      " << bins.count << '\n';

    const MLSSbinary* bin = findNonRelocMainKernel(bins.data, bins.count);
    if (bin == nullptr)
    {
        std::cerr << "No non-relocatable Conv MxN binary found\n";
        return EXIT_FAILURE;
    }

    std::cout << "Selected bin:  kernel='" << (bin->m_pKernelName ? bin->m_pKernelName : "?")
              << "' ASIC=" << (bin->m_ASIC ? bin->m_ASIC : "?")
              << " grid={" << bin->m_grid.m_x << ',' << bin->m_grid.m_y << ',' << bin->m_grid.m_z << "}"
              << " size=" << bin->m_binarySize << " bytes\n";

    const std::string runtimeGfx = props.gcnArchName;
    std::vector<std::uint8_t> moduleImage = prepareModuleImage(*bin, runtimeGfx);

    hipModule_t module{};
    CHECK_HIP(hipModuleLoadData(&module, moduleImage.data()));

    hipFunction_t kernel{};
    CHECK_HIP(hipModuleGetFunction(&kernel, module, "main"));
    std::cout << "hipModuleLoadData + hipModuleGetFunction(\"main\") OK\n";

    const std::size_t inputCount = static_cast<std::size_t>(n) * c * h * w;
    const std::size_t filterCount = static_cast<std::size_t>(k) * r * s * (c / groups);
    const std::size_t biasCount = hasBias ? static_cast<std::size_t>(k) : 0U;
    const std::size_t outputCount = static_cast<std::size_t>(n) * outH * outW * k;

    std::vector<float> hostInput(inputCount);
    std::vector<float> hostFilter(filterCount);
    std::vector<float> hostBias(biasCount);
    std::vector<float> hostOutput(outputCount, 0.0F);
    std::vector<float> cpuOutput(outputCount, 0.0F);

    fillSimpleTestData(hostInput, hostFilter, hostBias);
    computeConvCpuRef(hostInput, hostFilter, hostBias, cpuOutput);

    std::cout << "Test data:     input[channel]=1..C, filter[k]=3x3 ones on channel (k%C), bias=+0.5/-0.5\n";
    std::cout << "CPU ref[0..7]: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(8U, cpuOutput.size()); ++i)
        std::cout << cpuOutput[i] << ' ';
    std::cout << '\n';

    void* devInput = nullptr;
    void* devFilter = nullptr;
    void* devBias = nullptr;
    void* devOutput = nullptr;
    void* devSync = nullptr;
    void* devAcc = nullptr;

    CHECK_HIP(hipMalloc(&devInput, sizeof(float) * hostInput.size()));
    CHECK_HIP(hipMalloc(&devFilter, sizeof(float) * hostFilter.size()));
    CHECK_HIP(hipMalloc(&devBias, sizeof(float) * hostBias.size()));
    CHECK_HIP(hipMalloc(&devOutput, sizeof(float) * hostOutput.size()));

    const std::uint32_t kNGroups =
        bin->m_grid.m_x / (static_cast<std::uint32_t>(n) * static_cast<std::uint32_t>(groups));
    const std::size_t syncBufBytes = static_cast<std::size_t>(kNGroups) * sizeof(std::uint32_t);
    const std::size_t accBufBytes = sizeof(float) * hostOutput.size();

    CHECK_HIP(hipMalloc(&devSync, syncBufBytes));
    CHECK_HIP(hipMalloc(&devAcc, accBufBytes));

    CHECK_HIP(hipMemcpy(devInput, hostInput.data(), sizeof(float) * hostInput.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(devFilter, hostFilter.data(), sizeof(float) * hostFilter.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(devBias, hostBias.data(), sizeof(float) * hostBias.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(devOutput, 0, sizeof(float) * hostOutput.size()));
    CHECK_HIP(hipMemset(devSync, 0, syncBufBytes));
    CHECK_HIP(hipMemset(devAcc, 0, accBufBytes));

    const std::uint64_t flags64 =
        (hasBias ? (1ULL << 7) : 0ULL) | (1ULL << 9) | (1ULL << 14) | (1ULL << 15);

    std::unordered_map<std::string, KernelArg> argMap;
    argMap.emplace("n", KernelArg(static_cast<std::uint32_t>(n)));
    argMap.emplace("c", KernelArg(static_cast<std::uint32_t>(c)));
    argMap.emplace("h", KernelArg(static_cast<std::uint32_t>(h)));
    argMap.emplace("w", KernelArg(static_cast<std::uint32_t>(w)));
    argMap.emplace("k", KernelArg(static_cast<std::uint32_t>(k)));
    argMap.emplace("nGroups", KernelArg(kNGroups));
    argMap.emplace("flags64", KernelArg(flags64));
    argMap.emplace("dataAddr", KernelArg(devInput));
    argMap.emplace("filterAddr", KernelArg(devFilter));
    argMap.emplace("outputAddr", KernelArg(devOutput));
    argMap.emplace("reserved3", KernelArg(std::uint64_t{0}));
    argMap.emplace("r", KernelArg(static_cast<std::uint32_t>(r)));
    argMap.emplace("s", KernelArg(static_cast<std::uint32_t>(s)));
    argMap.emplace("padH", KernelArg(static_cast<std::int32_t>(startPadY)));
    argMap.emplace("padW", KernelArg(static_cast<std::int32_t>(startPadX)));
    argMap.emplace("outH", KernelArg(static_cast<std::uint32_t>(outH)));
    argMap.emplace("outW", KernelArg(static_cast<std::uint32_t>(outW)));
    argMap.emplace("biasAddr", KernelArg(devBias));
    argMap.emplace("alpha", KernelArg(1.0F));
    argMap.emplace("beta", KernelArg(0.0F));
    argMap.emplace("dOffset", KernelArg(static_cast<std::uint64_t>(dOffset)));
    argMap.emplace("fOffset", KernelArg(static_cast<std::uint64_t>(fOffset)));
    argMap.emplace("oOffset", KernelArg(static_cast<std::uint64_t>(oOffset)));
    argMap.emplace("bOffset", KernelArg(static_cast<std::uint64_t>(bOffset)));
    argMap.emplace("dNStride", KernelArg(static_cast<std::uint32_t>(dNStride)));
    argMap.emplace("dCStride", KernelArg(static_cast<std::uint32_t>(dCStride)));
    argMap.emplace("dHStride", KernelArg(static_cast<std::uint32_t>(dHStride)));
    argMap.emplace("reserved4", KernelArg(std::uint32_t{0}));
    argMap.emplace("fKStride", KernelArg(static_cast<std::uint32_t>(fKStride)));
    argMap.emplace("fCStride", KernelArg(static_cast<std::uint32_t>(fCStride)));
    argMap.emplace("fRStride", KernelArg(static_cast<std::uint32_t>(fRStride)));
    argMap.emplace("reserved5", KernelArg(std::uint32_t{0}));
    argMap.emplace("oNStride", KernelArg(static_cast<std::uint32_t>(oNStride)));
    argMap.emplace("oKStride", KernelArg(static_cast<std::uint32_t>(oKStride)));
    argMap.emplace("oHStride", KernelArg(static_cast<std::uint32_t>(oHStride)));
    argMap.emplace("reserved6", KernelArg(std::uint32_t{0}));
    argMap.emplace("G", KernelArg(static_cast<std::uint32_t>(groups)));
    argMap.emplace("dGStride", KernelArg(std::uint32_t{0}));
    argMap.emplace("fGStride", KernelArg(std::uint32_t{0}));
    argMap.emplace("oGStride", KernelArg(std::uint32_t{0}));
    argMap.emplace("activationMode", KernelArg(kKernelActivationRelu));
    argMap.emplace("syncLimit", KernelArg(static_cast<std::uint8_t>(255)));
    argMap.emplace("syncPeriod", KernelArg(static_cast<std::uint8_t>(0)));
    argMap.emplace("reserved8", KernelArg(static_cast<std::uint8_t>(0)));
    argMap.emplace("reserved9", KernelArg(static_cast<std::uint32_t>(0)));
    argMap.emplace("syncAddr", KernelArg(devSync));
    argMap.emplace("accAddr", KernelArg(devAcc));
    argMap.emplace("aOffset", KernelArg(static_cast<std::uint64_t>(0)));

    const std::vector<KernelArg> args = buildArgsFromBinary(*bin, argMap);
    if (args.empty())
        return EXIT_FAILURE;

    std::vector<std::byte> kernArgs;
    KernelArg::serialize(args, kernArgs);

    std::size_t kernArgSize = kernArgs.size();
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernArgs.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &kernArgSize,
        HIP_LAUNCH_PARAM_END};

    const unsigned int gridX = bin->m_grid.m_x;
    const unsigned int gridY = bin->m_grid.m_y;
    const unsigned int gridZ = bin->m_grid.m_z;

    std::cout << "Launch:        grid={" << gridX << ',' << gridY << ',' << gridZ << "}"
              << " block={" << kBlockDimX << ",1,1}"
              << " nGroups=" << kNGroups
              << " kernargBytes=" << kernArgSize << '\n';

    CHECK_HIP(hipModuleLaunchKernel(
        kernel,
        gridX,
        gridY,
        gridZ,
        kBlockDimX,
        1U,
        1U,
        static_cast<unsigned int>(bin->m_sharedMemInBytes),
        nullptr,
        nullptr,
        extra));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(hostOutput.data(), devOutput, sizeof(float) * hostOutput.size(), hipMemcpyDeviceToHost));

    std::size_t nonZeroCount = 0U;
    float maxAbs = 0.0F;
    for (float value : hostOutput)
    {
        if (value != 0.0F)
            ++nonZeroCount;
        maxAbs = std::max(maxAbs, std::fabs(value));
    }

    std::cout << "GPU out[0..7]: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(8U, hostOutput.size()); ++i)
        std::cout << hostOutput[i] << ' ';
    std::cout << '\n';

    std::cout << "nonZeroCount=" << nonZeroCount << " / " << hostOutput.size()
              << " maxAbs=" << maxAbs << '\n';

    CHECK_HIP(hipFree(devInput));
    CHECK_HIP(hipFree(devFilter));
    CHECK_HIP(hipFree(devBias));
    CHECK_HIP(hipFree(devOutput));
    CHECK_HIP(hipFree(devSync));
    CHECK_HIP(hipFree(devAcc));
    CHECK_HIP(hipModuleUnload(module));

    if (nonZeroCount == 0U)
    {
        std::cerr << "FAIL: output buffer is all zeros after launch\n";
        return EXIT_FAILURE;
    }

    if (!compareOutputs(hostOutput, cpuOutput, kCompareTolerance))
    {
        std::cerr << "FAIL: GPU output does not match CPU reference\n";
        return EXIT_FAILURE;
    }

    std::cout << "PASS: MIGraphX-style sample matches CPU on " << props.gcnArchName << '\n';
    return EXIT_SUCCESS;
}
