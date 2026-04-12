/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "utils.hpp"

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

} // namespace mlss::conv::utils
