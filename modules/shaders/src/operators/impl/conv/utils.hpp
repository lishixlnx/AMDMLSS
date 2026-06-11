/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"
#include "shaders/src/operators/impl/opUtils.hpp"

namespace mlss::conv::utils
{

struct GenericConvParams
{
    std::uint32_t w;
    std::uint32_t h;
    std::uint32_t c;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t s;
    std::uint32_t r;
    std::uint32_t outW;
    std::uint32_t outH;
    std::uint32_t dilationX;
    std::uint32_t dilationY;
    std::uint32_t startPadX;
    std::uint32_t startPadY;
    std::uint32_t endPadX;
    std::uint32_t endPadY;
    std::uint32_t outPadX;
    std::uint32_t outPadY;
    std::uint32_t convStrideX;
    std::uint32_t convStrideY;
    std::uint32_t inputStrideX;
    std::uint32_t inputStrideY;
    std::uint32_t filterStrideX;
    std::uint32_t filterStrideY;
    std::uint32_t groups;
    bool          hasBias;
    bool          crossCorrelation;
    bool          backward;

    std::uint32_t dNStride;
    std::uint32_t dHStride;
    std::uint32_t dCStride;
    std::uint32_t fKStride;
    std::uint32_t fCStride;
    std::uint32_t fRStride;
    std::uint32_t fSStride;
    std::uint32_t oNStride;
    std::uint32_t oHStride;
    std::uint32_t oKStride;
    std::uint64_t dOffset;
    std::uint64_t oOffset;
    std::uint64_t fOffset;
    std::uint64_t bOffset;

    DataTypeFlags              dataType;
    PrecisionFlags          precision;
    ActivationFunctionFlags activation;
};

GenericConvParams buildConvParams(const std::vector<Attribute>& attributes);


struct MisaConvArgs
{
    std::uint32_t hi;
    std::uint32_t wi;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t c;
    std::uint32_t g;
    std::uint32_t ho;
    std::uint32_t wo;
    std::uint32_t inStrideN;
    std::uint32_t inStrideC;
    std::uint32_t inStrideH;
    std::uint32_t inStrideW;
    std::uint32_t outStrideN;
    std::uint32_t outStrideK;
    std::uint32_t outStrideH;
    std::uint32_t outStrideW;
    std::uint32_t strideHw;
    std::uint32_t dilationHw;
    std::uint32_t padHw;
    std::uint32_t weiHw;
    std::uint32_t moveSliceK;
    std::uint32_t magic0;
    std::uint32_t magic1;
    std::uint32_t magic2;
    std::uint32_t magic3;
    std::uint32_t magic4;
    std::uint32_t magic5;
    std::uint32_t shiftPack0;
    std::uint32_t shiftPack1;
};

MisaConvArgs buildMisaConvArgs(const GenericConvParams& params, const std::uint32_t& tile_k);

MLSSdim3 MisaConvGetGridSize(
    const MisaConvArgs&  args, 
    const std::array<std::uint32_t, 3>& macroTile);

} // namespace mlss::conv::utils