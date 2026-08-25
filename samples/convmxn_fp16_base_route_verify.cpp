/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Verify fp16 Conv MxN queries return Winograd Base non-reloc / "main"
 * instead of Winograd Fury / "_amdgpu_cs_main".
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <amdmlss/amdmlss_api.h>

namespace
{
constexpr MLSSuint32 kW = 112U;
constexpr MLSSuint32 kH = 112U;
constexpr MLSSuint32 kC = 64U;
constexpr MLSSuint32 kN = 1U;
constexpr MLSSuint32 kK = 128U;
constexpr MLSSuint32 kR = 3U;
constexpr MLSSuint32 kS = 3U;
constexpr MLSSuint32 kOutW = 112U;
constexpr MLSSuint32 kOutH = 112U;
constexpr MLSSuint32 kGroups = 1U;
constexpr MLSSsize kMinBaseNonRelocSize = 50000U;

bool configureFp16ConvContext(MLSScontext* ctx)
{
    MLSSstring opName = const_cast<MLSSstring>(MLSS_CONV);
    if (mlssCreateContext(ctx, const_cast<MLSSstring>(MLSS_GFX1100), opName) != MLSS_SUCCESS)
    {
        return false;
    }

    MLSSuint32 w = kW;
    MLSSuint32 h = kH;
    MLSSuint32 c = kC;
    MLSSuint32 n = kN;
    MLSSuint32 k = kK;
    MLSSuint32 r = kR;
    MLSSuint32 s = kS;
    MLSSuint32 outW = kOutW;
    MLSSuint32 outH = kOutH;
    MLSSuint32 dilationX = 1U;
    MLSSuint32 dilationY = 1U;
    MLSSuint32 startPadX = 1U;
    MLSSuint32 startPadY = 1U;
    MLSSuint32 endPadX = 1U;
    MLSSuint32 endPadY = 1U;
    MLSSuint32 outPadX = 0U;
    MLSSuint32 outPadY = 0U;
    MLSSuint32 convStrideX = 1U;
    MLSSuint32 convStrideY = 1U;
    MLSSuint32 inputStrideX = 1U;
    MLSSuint32 inputStrideY = 1U;
    MLSSuint32 filterStrideX = 1U;
    MLSSuint32 filterStrideY = 1U;
    MLSSuint32 groups = kGroups;
    MLSSuint32 dNStride = kC * kH * kW;
    MLSSuint32 dHStride = kW;
    MLSSuint32 dCStride = kH * kW;
    MLSSuint32 fKStride = kC * kR * kS;
    MLSSuint32 fCStride = kR * kS;
    MLSSuint32 fRStride = kS;
    MLSSuint32 fSStride = 1U;
    MLSSuint32 oNStride = kK * kOutH * kOutW;
    MLSSuint32 oHStride = kOutW;
    MLSSuint32 oKStride = kOutH * kOutW;
    MLSSuint32 dOffset = 0U;
    MLSSuint32 oOffset = 0U;
    MLSSuint32 fOffset = 0U;
    MLSSuint32 bOffset = 0U;
    MLSSenum dataType = MLSS_FLOAT16;
    MLSSenum precision = MLSS_PRECISION_FLOAT16_ADD_FLOAT32;
    MLSSenum activation = MLSS_ACTIVATION_IDENTITY;
    MLSSbool hasBias = static_cast<MLSSbool>(false);
    MLSSbool crossCorrelation = static_cast<MLSSbool>(false);
    MLSSbool backward = static_cast<MLSSbool>(false);

#define SET_PARAM(attr, value)                                                                     \
    do                                                                                             \
    {                                                                                              \
        if (mlssSetParameterByEnum(ctx, opName, (attr), &(value)) != MLSS_SUCCESS)              \
        {                                                                                          \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

    SET_PARAM(MLSS_ATTR_CONV_W, w);
    SET_PARAM(MLSS_ATTR_CONV_H, h);
    SET_PARAM(MLSS_ATTR_CONV_C, c);
    SET_PARAM(MLSS_ATTR_CONV_N, n);
    SET_PARAM(MLSS_ATTR_CONV_K, k);
    SET_PARAM(MLSS_ATTR_CONV_S, s);
    SET_PARAM(MLSS_ATTR_CONV_R, r);
    SET_PARAM(MLSS_ATTR_CONV_OUTW, outW);
    SET_PARAM(MLSS_ATTR_CONV_OUTH, outH);
    SET_PARAM(MLSS_ATTR_CONV_DILATIONX, dilationX);
    SET_PARAM(MLSS_ATTR_CONV_DILATIONY, dilationY);
    SET_PARAM(MLSS_ATTR_CONV_STARTPADX, startPadX);
    SET_PARAM(MLSS_ATTR_CONV_STARTPADY, startPadY);
    SET_PARAM(MLSS_ATTR_CONV_ENDPADX, endPadX);
    SET_PARAM(MLSS_ATTR_CONV_ENDPADY, endPadY);
    SET_PARAM(MLSS_ATTR_CONV_OUTPADX, outPadX);
    SET_PARAM(MLSS_ATTR_CONV_OUTPADY, outPadY);
    SET_PARAM(MLSS_ATTR_CONV_CONVSTRIDEX, convStrideX);
    SET_PARAM(MLSS_ATTR_CONV_CONVSTRIDEY, convStrideY);
    SET_PARAM(MLSS_ATTR_CONV_INPUTSTRIDEX, inputStrideX);
    SET_PARAM(MLSS_ATTR_CONV_INPUTSTRIDEY, inputStrideY);
    SET_PARAM(MLSS_ATTR_CONV_FILTERSTRIDEX, filterStrideX);
    SET_PARAM(MLSS_ATTR_CONV_FILTERSTRIDEY, filterStrideY);
    SET_PARAM(MLSS_ATTR_CONV_GROUPS, groups);
    SET_PARAM(MLSS_ATTR_CONV_HASBIAS, hasBias);
    SET_PARAM(MLSS_ATTR_CONV_CROSSCORRELATION, crossCorrelation);
    SET_PARAM(MLSS_ATTR_CONV_BACKWARD, backward);
    SET_PARAM(MLSS_ATTR_CONV_DNSTRIDE, dNStride);
    SET_PARAM(MLSS_ATTR_CONV_DHSTRIDE, dHStride);
    SET_PARAM(MLSS_ATTR_CONV_DCSTRIDE, dCStride);
    SET_PARAM(MLSS_ATTR_CONV_FKSTRIDE, fKStride);
    SET_PARAM(MLSS_ATTR_CONV_FCSTRIDE, fCStride);
    SET_PARAM(MLSS_ATTR_CONV_FRSTRIDE, fRStride);
    SET_PARAM(MLSS_ATTR_CONV_FSSTRIDE, fSStride);
    SET_PARAM(MLSS_ATTR_CONV_ONSTRIDE, oNStride);
    SET_PARAM(MLSS_ATTR_CONV_OHSTRIDE, oHStride);
    SET_PARAM(MLSS_ATTR_CONV_OKSTRIDE, oKStride);
    SET_PARAM(MLSS_ATTR_CONV_DOFFSET, dOffset);
    SET_PARAM(MLSS_ATTR_CONV_OOFFSET, oOffset);
    SET_PARAM(MLSS_ATTR_CONV_FOFFSET, fOffset);
    SET_PARAM(MLSS_ATTR_CONV_BOFFSET, bOffset);
    SET_PARAM(MLSS_ATTR_CONV_DATATYPE, dataType);
    SET_PARAM(MLSS_ATTR_CONV_PRECISION, precision);
    SET_PARAM(MLSS_ATTR_CONV_ACTIVATION, activation);

#undef SET_PARAM

    return true;
}

} // namespace

int main()
{
    mlssSetVerboseLevel(MLSS_VERBOSE_NONE);

    MLSScontext ctx = 0;
    if (!configureFp16ConvContext(&ctx))
    {
        std::cerr << "configureFp16ConvContext failed\n";
        return EXIT_FAILURE;
    }

    MLSSstatus* statuses = nullptr;
    MLSSsize statusCount = 0;
    if (mlssGetCaps(ctx, &statuses, &statusCount) != MLSS_SUCCESS)
    {
        std::cerr << "mlssGetCaps failed\n";
        return EXIT_FAILURE;
    }

    MLSSbinary* bins = nullptr;
    MLSSsize binCount = 0;
    if (mlssGetBinariesEx(ctx, &bins, &binCount, MLSS_BINARY_KIND_NON_RELOCATABLE) != MLSS_SUCCESS
        || bins == nullptr || binCount == 0)
    {
        std::cerr << "mlssGetBinariesEx returned no non-reloc binaries\n";
        return EXIT_FAILURE;
    }

    std::cout << "fp16 conv non-reloc binaries: " << binCount << '\n';
    for (MLSSsize i = 0; i < binCount; ++i)
    {
        const char* kernel = bins[i].m_pKernelName ? bins[i].m_pKernelName : "?";
        std::cout << "  bin[" << i << "] kernel=" << kernel
                  << " reloc=" << (bins[i].m_isRelocatable ? 1 : 0)
                  << " size=" << bins[i].m_binarySize << '\n';
    }

    const MLSSbinary* selected = nullptr;
    for (MLSSsize i = 0; i < binCount; ++i)
    {
        if (!bins[i].m_isRelocatable && bins[i].m_pKernelName != nullptr
            && std::strcmp(bins[i].m_pKernelName, "main") == 0)
        {
            selected = &bins[i];
            break;
        }
    }

    if (selected == nullptr)
    {
        std::cerr << "FAIL: no non-reloc Winograd Base bin with kernel \"main\"\n";
        return EXIT_FAILURE;
    }

    if (selected->m_binarySize < kMinBaseNonRelocSize)
    {
        std::cerr << "FAIL: selected bin size " << selected->m_binarySize
                  << " looks like Fury, expected Base non-reloc (>= " << kMinBaseNonRelocSize
                  << " bytes)\n";
        return EXIT_FAILURE;
    }

    std::cout << "PASS: fp16 conv routed to Winograd Base non-reloc / main (size="
              << selected->m_binarySize << " bytes)\n";
    return EXIT_SUCCESS;
}
