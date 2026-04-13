/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "utils.hpp"
#include "shaders/shaders.hpp"

namespace mlss::conv::utils
{

GenericConvParams buildConvParams(const std::vector<Attribute>& attributes)
{
    GenericConvParams params{};

    for (const auto& attribute : attributes)
    {
        if (attribute.is(MLSS_ATTR_CONV_W))
        {
            params.w = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_H))
        {
            params.h = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_C))
        {
            params.c = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_N))
        {
            params.n = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_K))
        {
            params.k = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_S))
        {
            params.s = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_R))
        {
            params.r = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OUTW))
        {
            params.outW = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OUTH))
        {
            params.outH = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_STARTPADX))
        {
            params.startPadX = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_STARTPADY))
        {
            params.startPadY = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_ENDPADX))
        {
            params.endPadX = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_ENDPADY))
        {
            params.endPadY = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OUTPADX))
        {
            params.outPadX = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OUTPADY))
        {
            params.outPadY = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_CONVSTRIDEX))
        {
            params.convStrideX = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_CONVSTRIDEY))
        {
            params.convStrideY = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_INPUTSTRIDEX))
        {
            params.inputStrideX = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_INPUTSTRIDEY))
        {
            params.inputStrideY = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FILTERSTRIDEX))
        {
            params.filterStrideX = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FILTERSTRIDEY))
        {
            params.filterStrideY = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_GROUPS))
        {
            params.groups = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_HASBIAS))
        {
            params.hasBias = attribute.value<bool>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_CROSSCORRELATION))
        {
            params.crossCorrelation = attribute.value<bool>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_BACKWARD))
        {
            params.backward = attribute.value<bool>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_DNSTRIDE))
        {
            params.dNStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_DHSTRIDE))
        {
            params.dHStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_DCSTRIDE))
        {
            params.dCStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FKSTRIDE))
        {
            params.fKStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FCSTRIDE))
        {
            params.fCStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FRSTRIDE))
        {
            params.fRStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FSSTRIDE))
        {
            params.fSStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_ONSTRIDE))
        {
            params.oNStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OHSTRIDE))
        {
            params.oHStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OKSTRIDE))
        {
            params.oKStride = attribute.value<std::uint32_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_DOFFSET))
        {
            params.dOffset = attribute.value<std::uint64_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_OOFFSET))
        {
            params.oOffset = attribute.value<std::uint64_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_FOFFSET))
        {
            params.fOffset = attribute.value<std::uint64_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_BOFFSET))
        {
            params.bOffset = attribute.value<std::uint64_t>();
        }
        else if (attribute.is(MLSS_ATTR_CONV_DATATYPE))
        {
            params.dataType = static_cast<DataTypeFlags>(attribute.value<std::uint64_t>());
        }
        else if (attribute.is(MLSS_ATTR_CONV_PRECISION))
        {
            params.precision = static_cast<PrecisionFlags>(attribute.value<std::uint32_t>());
        }
        else if (attribute.is(MLSS_ATTR_CONV_ACTIVATION))
        {
            params.activation = static_cast<ActivationFunctionFlags>(attribute.value<std::uint32_t>());
        }
    }

    return params;
}

namespace
{
constexpr std::uint32_t VectorC = 0x08;

// =====================================================================================================================
std::uint32_t GetMoveSliceK(
    const GenericConvParams& params, 
    const std::uint32_t& tile_k)
{
    std::uint32_t cPerGPerV = (params.c / params.groups) / VectorC;
    std::uint32_t gemmKLeft = tile_k / VectorC;
    std::uint32_t sMoveSliceKC = gemmKLeft % cPerGPerV;
    gemmKLeft /= cPerGPerV;
    std::uint32_t sMoveSliceKX = gemmKLeft % params.s;
    std::uint32_t sMoveSliceKY = (gemmKLeft / params.s) % params.r;
    return (sMoveSliceKY << 16) | (sMoveSliceKX << 8) | sMoveSliceKC;
}

struct MagicDivU32
{
    std::uint32_t magic;
    std::uint8_t shift;
};

MagicDivU32 MagicDivU32Gen(
    std::uint32_t d)
{
    MLSS_ASSERT((d >= 1) && (d <= INT32_MAX));
    std::uint8_t shift;
    for (shift = 0; shift < 32; shift++)
    {
        if ((1ull << shift) >= d)
        {
            break;
        }
    }

    const std::uint64_t magic = (((1ull << 32) * ((1ull << shift) - d)) / d) + 1;
    MLSS_ASSERT(magic <= INT32_MAX); // 0xfffffffful

    MagicDivU32 result;
    result.magic = static_cast<decltype(result.magic)>(magic);
    result.shift = shift;
    return result;
}

// =====================================================================================================================
// Derive packed shift values for magic numbers required in the shader argument list
std::uint32_t MagicDivU32PackShift(
    std::uint8_t s0,
    std::uint8_t s1,
    std::uint8_t s2,
    std::uint8_t s3)
{
    const std::uint32_t shift0 = static_cast<std::uint32_t>(s0);
    const std::uint32_t shift1 = static_cast<std::uint32_t>(s1);
    const std::uint32_t shift2 = static_cast<std::uint32_t>(s2);
    const std::uint32_t shift3 = static_cast<std::uint32_t>(s3);
    return (shift3 << 24) | (shift2 << 16) | (shift1 << 8) | shift0;
}

// =====================================================================================================================
// Compute batch split size based on configuration parameters
std::uint32_t ConvSplitBatchSize(
    const MisaConvArgs& args)
{
    const std::uint64_t memorySizeInput = args.c * args.hi * args.wi * sizeof(std::uint16_t);
    const std::uint64_t memorySizeOutput = args.k * args.ho * args.wo * sizeof(std::uint16_t);
    constexpr std::uint64_t Size4GbMinusOne = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
    const std::uint32_t n = args.n;
    const std::uint64_t imageSize =
        memorySizeInput >= memorySizeOutput ? memorySizeInput : memorySizeOutput;
    std::uint32_t splitedN = std::uint32_t(Size4GbMinusOne / imageSize);
    MLSS_ASSERT(splitedN != 0);
    while (splitedN >= 1)
    {
        if ((n % splitedN == 0) && ((splitedN * imageSize) < Size4GbMinusOne))
        {
            break;
        }

        splitedN--;
    }
    MLSS_ASSERT(((splitedN * imageSize) < Size4GbMinusOne) && ((n % splitedN) == 0));
    return n / splitedN;
}

// =====================================================================================================================
// Compute array of grid sizes used as Dispatch arguments
MLSSdim3 GetGridSize(
    const MisaConvArgs&  args, 
    const std::array<std::uint32_t, 3>& macroTile)
{
    MLSSdim3 grid{};
    const std::uint32_t splits = ConvSplitBatchSize(args);
    const std::uint32_t gemmM = args.k / args.g;
    const std::uint32_t gemmN = (args.n / splits) * args.ho * args.wo;

    grid.m_x = (gemmN + macroTile[1] - 1) / macroTile[1];
    grid.m_y = (gemmM + macroTile[0] - 1) / macroTile[0];
    grid.m_z = splits * args.g;
    return grid;
}

} // namespace

MLSSdim3 MisaConvGetGridSize(
    const MisaConvArgs&  args,
    const std::array<std::uint32_t, 3>& macroTile)
{
    return GetGridSize(args, macroTile);
}

MisaConvArgs buildMisaConvArgs(const GenericConvParams& params, const std::uint32_t& tile_k)
{
    MisaConvArgs args{};

    args.hi = params.h;
    args.wi = params.w;
    args.n = params.n;
    args.k = params.k;
    args.c = params.c;
    args.g = params.groups;
    args.ho = params.outH;
    args.wo = params.outW;
    args.inStrideN  = sizeof(std::uint16_t) * params.c * params.h * params.w;
    args.inStrideC  = sizeof(std::uint16_t) * VectorC;
    args.inStrideH  = sizeof(std::uint16_t) * params.c * params.w;
    args.inStrideW  = sizeof(std::uint16_t) * params.c;
    args.outStrideN = sizeof(std::uint16_t) * params.k * params.outH * params.outW;
    args.outStrideK = sizeof(std::uint16_t) * VectorC;
    args.outStrideH = sizeof(std::uint16_t) * params.k * params.outW;
    args.outStrideW = sizeof(std::uint16_t) * params.k;
    args.strideHw   = (params.convStrideY << 16) | params.convStrideX;
    args.dilationHw = (params.filterStrideY << 16) | params.filterStrideX;
    args.padHw      = (params.startPadY << 16) | params.startPadX;
    args.weiHw      = (params.r << 16) | params.s;
    args.moveSliceK = GetMoveSliceK(params, tile_k);

    const MagicDivU32 mdiv0 = MagicDivU32Gen(params.groups);
    const std::uint32_t cPergPerV = params.c / params.groups / VectorC;
    const MagicDivU32 mdiv1 = MagicDivU32Gen(cPergPerV);
    const MagicDivU32 mdiv2 = MagicDivU32Gen(params.outH);
    const MagicDivU32 mdiv3 = MagicDivU32Gen(params.outW);
    const MagicDivU32 mdiv4 = MagicDivU32Gen(params.r);
    const MagicDivU32 mdiv5 = MagicDivU32Gen(params.s);

    args.magic0 = mdiv0.magic;
    args.magic1 = mdiv1.magic;
    args.magic2 = mdiv2.magic;
    args.magic3 = mdiv3.magic;
    args.magic4 = mdiv4.magic;
    args.magic5 = mdiv5.magic;
    args.shiftPack0 = MagicDivU32PackShift(mdiv0.shift, mdiv1.shift, mdiv2.shift, mdiv3.shift);
    args.shiftPack1 = MagicDivU32PackShift(mdiv4.shift, mdiv5.shift, 0, 0);

    return args;
}

} // namespace mlss::conv::utils
