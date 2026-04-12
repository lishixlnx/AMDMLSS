#pragma once

#include "core/core.hpp"
#include "opUtils.hpp"

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

} // namespace mlss::conv::utils