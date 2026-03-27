/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "core/core.hpp"

#include <format>

namespace mlss
{

    //=====================================================================================================================
    constexpr const std::size_t infoBufferSize = 0x4000;

    //=====================================================================================================================
    constexpr const char* getMLSSTypesList()
    {
        return "[booltype,\n"
               "int4,\n"
               "uint4,\n"
               "int8,\n"
               "uint8,\n"
               "int16,\n"
               "uint16,\n"
               "int32,\n"
               "uint32,\n"
               "int64,\n"
               "uint64,\n"
               "float4,\n"
               "float8,\n"
               "float16,\n"
               "float32,\n"
               "float64,\n"
               "bfloat4,\n"
               "bfloat8,\n"
               "bfloat16,\n"
               "enum,\n"
               "array,\n"
               "list]\n";
    }

    //=====================================================================================================================
    constexpr const char* getMLSSActivationFunctionList()
    {
        return "[elu,\n"
               "hardmax,\n"
               "hard_sigmoid,\n"
               "identity,\n"
               "leaky_relu,\n"
               "linear,\n"
               "log_softmax,\n"
               "parameterized_relu,\n"
               "parametric_softplus,\n"
               "relu,\n"
               "scaled_elu,\n"
               "scaled_tanh,\n"
               "sigmoid,\n"
               "softmax,\n"
               "softplus,\n"
               "softsign,\n"
               "tanh,\n"
               "thresholded_relu]\n";
    }

    //=====================================================================================================================
    constexpr const char* getMLSSPrecisionList()
    {
        return "[prec_fp16,\n"
               "prec_fp32,\n"
               "prec_fp16_add_fp32]\n";
    }

    //=====================================================================================================================
    template <OperatorFlag flag>
    constexpr const char* getMLSSOpSupportedTypeList()
    {
        if constexpr (flag == OperatorFlag::GQA)
        {
            return "[MLSS_FLOAT16]";
        }
        else if constexpr (flag == OperatorFlag::MHA)
        {
            return "[MLSS_FLOAT16]";
        }
        else if constexpr (flag == OperatorFlag::GEMM)
        {
            return "[MLSS_FLOAT16]";
        }
        else if constexpr (flag == OperatorFlag::QUANTIZED_GEMM)
        {
            return "[MLSS_FLOAT16]";
        }
        else if constexpr (flag == OperatorFlag::CONV)
        {
            return "[MLSS_FLOAT16, MLSS_FLOAT32]";
        }
        else if constexpr (flag == OperatorFlag::CONV_DILATED)
        {
            return "[MLSS_FLOAT16]";
        }
        else if constexpr (flag == OperatorFlag::MVN)
        {
            return "[MLSS_FLOAT16, MLSS_FLOAT32]";
        }
        else
        {
            return nullptr;
        }
    }

    //=====================================================================================================================
    std::string getAttributeNameFromEnum(const std::uint32_t& flag)
    {
        std::string ret;

        switch (flag)
        {
            case MLSS_ATTR_GQA_BATCH:
                ret = "batchSize";
                break;

            case MLSS_ATTR_GQA_QSEQ:
                ret = "qSeqLength";
                break;

            case MLSS_ATTR_GQA_KVSEQ:
                ret = "kvSeqLength";
                break;

            case MLSS_ATTR_GQA_KDIM:
                ret = "kDim";
                break;

            case MLSS_ATTR_GQA_VDIM:
                ret = "vDim";
                break;

            case MLSS_ATTR_GQA_SIZEHEADS:
                ret = "sizeHeads";
                break;

            case MLSS_ATTR_GQA_PACKING:
                ret = "packing";
                break;

            case MLSS_ATTR_GQA_QHEADCOUNT:
                ret = "qHeadCount";
                break;

            case MLSS_ATTR_GQA_KVHEADCOUNT:
                ret = "kvHeadCount";
                break;

            case MLSS_ATTR_GQA_PASTSEQLENGTHSSIZE:
                ret = "pastSeqLengthsSize";
                break;

            case MLSS_ATTR_GQA_SCALE:
                ret = "scale";
                break;

            case MLSS_ATTR_GQA_DATATYPE:
                ret = "dataType";
                break;

            case MLSS_ATTR_GQA_QSTRIDES:
                ret = "qStrides";
                break;

            case MLSS_ATTR_GQA_KSTRIDES:
                ret = "kStrides";
                break;

            case MLSS_ATTR_GQA_VSTRIDES:
                ret = "vStrides";
                break;

            case MLSS_ATTR_GQA_OUTPUTSTRIDES:
                ret = "outputStrides";
                break;

            case MLSS_ATTR_MHA_BATCH:
                ret = "batchSize";
                break;

            case MLSS_ATTR_MHA_QSEQ:
                ret = "qSeqLength";
                break;

            case MLSS_ATTR_MHA_KVSEQ:
                ret = "kvSeqLength";
                break;

            case MLSS_ATTR_MHA_KDIM:
                ret = "kDim";
                break;

            case MLSS_ATTR_MHA_VDIM:
                ret = "vDim";
                break;

            case MLSS_ATTR_MHA_SIZEHEADS:
                ret = "sizeHeads";
                break;

            case MLSS_ATTR_MHA_PACKING:
                ret = "packing";
                break;

            case MLSS_ATTR_MHA_HEADCOUNT:
                ret = "headCount";
                break;

            case MLSS_ATTR_MHA_SCALE:
                ret = "scale";
                break;

            case MLSS_ATTR_MHA_DATATYPE:
                ret = "dataType";
                break;

            case MLSS_ATTR_MHA_QSTRIDES:
                ret = "qStrides";
                break;

            case MLSS_ATTR_MHA_KSTRIDES:
                ret = "kStrides";
                break;

            case MLSS_ATTR_MHA_VSTRIDES:
                ret = "vStrides";
                break;

            case MLSS_ATTR_MHA_OUTPUTSTRIDES:
                ret = "outputStrides";
                break;

            case MLSS_ATTR_GEMM_M:
                ret = "m";
                break;

            case MLSS_ATTR_GEMM_N:
                ret = "n";
                break;

            case MLSS_ATTR_GEMM_K:
                ret = "k";
                break;

            case MLSS_ATTR_GEMM_BATCH:
                ret = "batchSize";
                break;

            case MLSS_ATTR_GEMM_ALPHA:
                ret = "alpha";
                break;

            case MLSS_ATTR_GEMM_BETA:
                ret = "beta";
                break;

            case MLSS_ATTR_GEMM_HASC:
                ret = "hasC";
                break;

            case MLSS_ATTR_GEMM_TRANSA:
                ret = "transA";
                break;

            case MLSS_ATTR_GEMM_TRANSB:
                ret = "transB";
                break;

            case MLSS_ATTR_GEMM_DATATYPE:
                ret = "dataType";
                break;

            case MLSS_ATTR_GEMM_PRECISION:
                ret = "precision";
                break;

            case MLSS_ATTR_GEMM_ACTIVATION:
                ret = "activation";
                break;

            case MLSS_ATTR_QGEMM_M:
                ret = "m";
                break;

            case MLSS_ATTR_QGEMM_N:
                ret = "n";
                break;

            case MLSS_ATTR_QGEMM_K:
                ret = "k";
                break;

            case MLSS_ATTR_QGEMM_BATCH:
                ret = "batchSize";
                break;

            case MLSS_ATTR_QGEMM_ALPHA:
                ret = "alpha";
                break;

            case MLSS_ATTR_QGEMM_BETA:
                ret = "beta";
                break;

            case MLSS_ATTR_QGEMM_HASC:
                ret = "hasC";
                break;

            case MLSS_ATTR_QGEMM_TRANSA:
                ret = "transA";
                break;

            case MLSS_ATTR_QGEMM_TRANSB:
                ret = "transB";
                break;

            case MLSS_ATTR_QGEMM_HASASCALE:
                ret = "hasAScale";
                break;

            case MLSS_ATTR_QGEMM_HASAZP:
                ret = "hasAZP";
                break;

            case MLSS_ATTR_QGEMM_HASBSCALE:
                ret = "hasBScale";
                break;

            case MLSS_ATTR_QGEMM_HASBZP:
                ret = "hasBZP";
                break;

            case MLSS_ATTR_QGEMM_HASCSACLE:
                ret = "hasCScale";
                break;

            case MLSS_ATTR_QGEMM_HASCZP:
                ret = "hasCZP";
                break;

            case MLSS_ATTR_QGEMM_HASOUTSCALE:
                ret = "hasOutScale";
                break;

            case MLSS_ATTR_QGEMM_HASOUTZP:
                ret = "hasOutZP";
                break;

            case MLSS_ATTR_QGEMM_QRATIOBSCALE:
                ret = "qRatioBScale";
                break;

            case MLSS_ATTR_QGEMM_QRATIOBZP:
                ret = "qRatioBZP";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEA:
                ret = "dataTypeA";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEB:
                ret = "dataTypeB";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEC:
                ret = "dataTypeC";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEOUTPUT:
                ret = "dataTypeOutput";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEQPARAMSA:
                ret = "dataTypeQParamsA";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEQPARAMSB:
                ret = "dataTypeQParamsB";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEQPARAMSC:
                ret = "dataTypeQParamsC";
                break;

            case MLSS_ATTR_QGEMM_DATATYPEQPARAMSOUTPUT:
                ret = "dataTypeQParamsOutput";
                break;

            case MLSS_ATTR_QGEMM_PRECISION:
                ret = "precision";
                break;

            case MLSS_ATTR_QGEMM_ACTIVATION:
                ret = "activation";
                break;

            case MLSS_ATTR_MVN_N:
                ret = "n";
                break;

            case MLSS_ATTR_MVN_C:
                ret = "c";
                break;

            case MLSS_ATTR_MVN_H:
                ret = "h";
                break;

            case MLSS_ATTR_MVN_W:
                ret = "w";
                break;

            case MLSS_ATTR_MVN_EPSILON:
                ret = "epsilon";
                break;

            case MLSS_ATTR_MVN_CROSSCHANNEL:
                ret = "crossChannel";
                break;

            case MLSS_ATTR_MVN_SBDIMS:
                ret = "sbDims";
                break;

            case MLSS_ATTR_MVN_HASSCALE:
                ret = "hasScale";
                break;

            case MLSS_ATTR_MVN_HASBIAS:
                ret = "hasBias";
                break;

            case MLSS_ATTR_MVN_DATATYPE:
                ret = "dataType";
                break;

            case MLSS_ATTR_MVN_ACTIVATION:
                ret = "activation";
                break;

            case MLSS_ATTR_CONV_W:
                ret = "w";
                break;

            case MLSS_ATTR_CONV_H:
                ret = "h";
                break;

            case MLSS_ATTR_CONV_C:
                ret = "c";
                break;

            case MLSS_ATTR_CONV_N:
                ret = "n";
                break;

            case MLSS_ATTR_CONV_K:
                ret = "k";
                break;

            case MLSS_ATTR_CONV_S:
                ret = "s";
                break;

            case MLSS_ATTR_CONV_R:
                ret = "r";
                break;

            case MLSS_ATTR_CONV_OUTW:
                ret = "outW";
                break;

            case MLSS_ATTR_CONV_OUTH:
                ret = "outH";
                break;

            case MLSS_ATTR_CONV_STARTPADX:
                ret = "startPadX";
                break;

            case MLSS_ATTR_CONV_STARTPADY:
                ret = "startPadY";
                break;

            case MLSS_ATTR_CONV_ENDPADX:
                ret = "endPadX";
                break;

            case MLSS_ATTR_CONV_ENDPADY:
                ret = "endPadY";
                break;

            case MLSS_ATTR_CONV_OUTPADX:
                ret = "outPadX";
                break;

            case MLSS_ATTR_CONV_OUTPADY:
                ret = "outPadY";
                break;

            case MLSS_ATTR_CONV_CONVSTRIDEX:
                ret = "convStrideX";
                break;

            case MLSS_ATTR_CONV_CONVSTRIDEY:
                ret = "convStrideY";
                break;

            case MLSS_ATTR_CONV_INPUTSTRIDEX:
                ret = "inputStrideX";
                break;

            case MLSS_ATTR_CONV_INPUTSTRIDEY:
                ret = "inputStrideY";
                break;

            case MLSS_ATTR_CONV_FILTERSTRIDEX:
                ret = "filterStrideX";
                break;

            case MLSS_ATTR_CONV_FILTERSTRIDEY:
                ret = "filterStrideY";
                break;

            case MLSS_ATTR_CONV_GROUPS:
                ret = "groups";
                break;

            case MLSS_ATTR_CONV_HASBIAS:
                ret = "hasBias";
                break;

            case MLSS_ATTR_CONV_CROSSCORRELATION:
                ret = "crossCorrelation";
                break;

            case MLSS_ATTR_CONV_BACKWARD:
                ret = "backward";
                break;

            case MLSS_ATTR_CONV_DNSTRIDE:
                ret = "dNStride";
                break;

            case MLSS_ATTR_CONV_DHSTRIDE:
                ret = "dHStride";
                break;

            case MLSS_ATTR_CONV_DCSTRIDE:
                ret = "dCStride";
                break;

            case MLSS_ATTR_CONV_FKSTRIDE:
                ret = "fKStride";
                break;

            case MLSS_ATTR_CONV_FCSTRIDE:
                ret = "fCStride";
                break;

            case MLSS_ATTR_CONV_FRSTRIDE:
                ret = "fRStride";
                break;

            case MLSS_ATTR_CONV_FSSTRIDE:
                ret = "fSStride";
                break;

            case MLSS_ATTR_CONV_ONSTRIDE:
                ret = "oNStride";
                break;

            case MLSS_ATTR_CONV_OHSTRIDE:
                ret = "oHStride";
                break;

            case MLSS_ATTR_CONV_OKSTRIDE:
                ret = "oKStride";
                break;

            case MLSS_ATTR_CONV_DOFFSET:
                ret = "dOffset";
                break;

            case MLSS_ATTR_CONV_OOFFSET:
                ret = "oOffset";
                break;

            case MLSS_ATTR_CONV_FOFFSET:
                ret = "fOffset";
                break;
            case MLSS_ATTR_CONV_BOFFSET:
                ret = "bOffset";
                break;

            case MLSS_ATTR_CONV_DATATYPE:
                ret = "dataType";
                break;

            case MLSS_ATTR_CONV_PRECISION:
                ret = "precision";
                break;
            case MLSS_ATTR_CONV_ACTIVATION:
                ret = "activation";
                break;

            case MLSS_ATTR_CONV_DILATED_W:
                ret = "w";
                break;

            case MLSS_ATTR_CONV_DILATED_H:
                ret = "h";
                break;

            case MLSS_ATTR_CONV_DILATED_C:
                ret = "c";
                break;

            case MLSS_ATTR_CONV_DILATED_N:
                ret = "n";
                break;

            case MLSS_ATTR_CONV_DILATED_K:
                ret = "k";
                break;

            case MLSS_ATTR_CONV_DILATED_S:
                ret = "s";
                break;

            case MLSS_ATTR_CONV_DILATED_R:
                ret = "r";
                break;

            case MLSS_ATTR_CONV_DILATED_OUTW:
                ret = "outW";
                break;

            case MLSS_ATTR_CONV_DILATED_OUTH:
                ret = "outH";
                break;

            case MLSS_ATTR_CONV_DILATED_STARTPADX:
                ret = "startPadX";
                break;

            case MLSS_ATTR_CONV_DILATED_STARTPADY:
                ret = "startPadY";
                break;

            case MLSS_ATTR_CONV_DILATED_ENDPADX:
                ret = "endPadX";
                break;

            case MLSS_ATTR_CONV_DILATED_ENDPADY:
                ret = "endPadY";
                break;

            case MLSS_ATTR_CONV_DILATED_OUTPADX:
                ret = "outPadX";
                break;

            case MLSS_ATTR_CONV_DILATED_OUTPADY:
                ret = "outPadY";
                break;

            case MLSS_ATTR_CONV_DILATED_CONVSTRIDEX:
                ret = "convStrideX";
                break;

            case MLSS_ATTR_CONV_DILATED_CONVSTRIDEY:
                ret = "convStrideY";
                break;

            case MLSS_ATTR_CONV_DILATED_INPUTSTRIDEX:
                ret = "inputStrideX";
                break;

            case MLSS_ATTR_CONV_DILATED_INPUTSTRIDEY:
                ret = "inputStrideY";
                break;

            case MLSS_ATTR_CONV_DILATED_FILTERSTRIDEX:
                ret = "filterStrideX";
                break;

            case MLSS_ATTR_CONV_DILATED_FILTERSTRIDEY:
                ret = "filterStrideY";
                break;

            case MLSS_ATTR_CONV_DILATED_GROUPS:
                ret = "groups";
                break;

            case MLSS_ATTR_CONV_DILATED_HASBIAS:
                ret = "hasBias";
                break;

            case MLSS_ATTR_CONV_DILATED_CROSSCORRELATION:
                ret = "crossCorrelation";
                break;

            case MLSS_ATTR_CONV_DILATED_BACKWARD:
                ret = "backward";
                break;

            case MLSS_ATTR_CONV_DILATED_DNSTRIDE:
                ret = "dNStride";
                break;

            case MLSS_ATTR_CONV_DILATED_DHSTRIDE:
                ret = "dHStride";
                break;

            case MLSS_ATTR_CONV_DILATED_DCSTRIDE:
                ret = "dCStride";
                break;

            case MLSS_ATTR_CONV_DILATED_FKSTRIDE:
                ret = "fKStride";
                break;

            case MLSS_ATTR_CONV_DILATED_FCSTRIDE:
                ret = "fCStride";
                break;

            case MLSS_ATTR_CONV_DILATED_FRSTRIDE:
                ret = "fRStride";
                break;

            case MLSS_ATTR_CONV_DILATED_FSSTRIDE:
                ret = "fSStride";
                break;

            case MLSS_ATTR_CONV_DILATED_ONSTRIDE:
                ret = "oNStride";
                break;

            case MLSS_ATTR_CONV_DILATED_OHSTRIDE:
                ret = "oHStride";
                break;

            case MLSS_ATTR_CONV_DILATED_OKSTRIDE:
                ret = "oKStride";
                break;

            case MLSS_ATTR_CONV_DILATED_DOFFSET:
                ret = "dOffset";
                break;

            case MLSS_ATTR_CONV_DILATED_OOFFSET:
                ret = "oOffset";
                break;

            case MLSS_ATTR_CONV_DILATED_FOFFSET:
                ret = "fOffset";
                break;

            case MLSS_ATTR_CONV_DILATED_BOFFSET:
                ret = "bOffset";
                break;

            case MLSS_ATTR_CONV_DILATED_DATATYPE:
                ret = "dataType";
                break;

            case MLSS_ATTR_CONV_DILATED_PRECISION:
                ret = "precision";
                break;

            case MLSS_ATTR_CONV_DILATED_ACTIVATION:
                ret = "activation";
                break;

            default:
                ret = "unknow attribute";
                break;
        }

        return ret;
    }

    //=====================================================================================================================
    template <class T>
    constexpr const char* getTypeNameAsString()
    {
        if constexpr (std::is_same<T, bool>::value)
        {
            return "bool";
        }
        else if constexpr (std::is_same<T, std::int8_t>::value)
        {
            return "int8";
        }
        else if constexpr (std::is_same<T, std::uint8_t>::value)
        {
            return "uint8";
        }
        else if constexpr (std::is_same<T, std::int16_t>::value)
        {
            return "int16";
        }
        else if constexpr (std::is_same<T, std::uint16_t>::value)
        {
            return "uint16";
        }
        else if constexpr (std::is_same<T, std::int32_t>::value)
        {
            return "int32";
        }
        else if constexpr (std::is_same<T, std::uint32_t>::value)
        {
            return "uint32";
        }
        else if constexpr (std::is_same<T, std::int64_t>::value)
        {
            return "int64";
        }
        else if constexpr (std::is_same<T, std::uint64_t>::value)
        {
            return "uint64";
        }
        else if constexpr (std::is_same<T, float>::value)
        {
            return "float32";
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            return "float64";
        }
        else
        {
            return "unknow type";
        }
    }

    //=================================================================================================================
    //                                   createAttributes
    //=================================================================================================================

    //=====================================================================================================================
    void createAttributes(const std::string& opName, std::vector<Attribute>& attributes)
    {

        constexpr const char* uint32_t_RANGE = "[1, 4294967295 (max uint32)]";
        constexpr const char* FLOAT32_RANGE = "[0.f, 1.f]";
        constexpr const char* BOOL_RANGE = "[0, 1]";
        constexpr const char* uint32_t_RANGE_x4 = "4 x [1, 4294967295 (max uint32)]";

        if (opName == MLSS_GQA)
        {
            attributes.reserve(16);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_BATCH, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_QSEQ, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_KVSEQ, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_KDIM, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_VDIM, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_SIZEHEADS, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>("[0, 2, 3]", MLSS_ATTR_GQA_PACKING, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_QHEADCOUNT, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_KVHEADCOUNT, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GQA_PASTSEQLENGTHSSIZE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_GQA_SCALE, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::GQA>(), MLSS_ATTR_GQA_DATATYPE, static_cast<std::uint32_t>(MLSS_ENUM)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_GQA_QSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_GQA_KSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_GQA_VSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_GQA_OUTPUTSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
        }
        else if (opName == MLSS_MHA)
        {
            attributes.reserve(14);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_BATCH, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_QSEQ, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_KVSEQ, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_KDIM, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_VDIM, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_SIZEHEADS, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>("[0, 2, 3]", MLSS_ATTR_MHA_PACKING, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MHA_HEADCOUNT, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_MHA_SCALE, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::MHA>(), MLSS_ATTR_MHA_DATATYPE, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_MHA_QSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_MHA_KSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_MHA_VSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_MHA_OUTPUTSTRIDES, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
        }
        else if (opName == MLSS_MVN)
        {
            attributes.reserve(12);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MVN_N, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MVN_C, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MVN_H, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_MVN_W, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_MVN_EPSILON, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_MVN_CROSSCHANNEL, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<std::array<uint32_t, 4>>(uint32_t_RANGE_x4, MLSS_ATTR_MVN_SBDIMS, makeArrayEnum(MLSS_ARRAY | MLSS_UINT32, 4)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_MVN_HASSCALE, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_MVN_HASBIAS, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::MVN>(), MLSS_ATTR_MVN_DATATYPE, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSActivationFunctionList(), MLSS_ATTR_MVN_ACTIVATION, MLSS_ENUM));
        }
        else if (opName == MLSS_CONV_DILATED)
        {
            attributes.reserve(36);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_W, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_H, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_C, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_N, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_K, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_S, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_R, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OUTW, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OUTH, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_STARTPADX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_STARTPADY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_ENDPADX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_ENDPADY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OUTPADX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OUTPADY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_CONVSTRIDEX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_CONVSTRIDEY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_INPUTSTRIDEX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_INPUTSTRIDEY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FILTERSTRIDEX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FILTERSTRIDEY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_GROUPS, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_CONV_DILATED_HASBIAS, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_CONV_DILATED_CROSSCORRELATION, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_CONV_DILATED_BACKWARD, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_DNSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_DHSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_DCSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FKSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FCSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FRSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FSSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_ONSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OHSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OKSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_DOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_OOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_FOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DILATED_BOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::CONV_DILATED>(), MLSS_ATTR_CONV_DILATED_DATATYPE, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSPrecisionList(), MLSS_ATTR_CONV_DILATED_PRECISION, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSActivationFunctionList(), MLSS_ATTR_CONV_DILATED_ACTIVATION, MLSS_ENUM));
        }
        else if (opName == MLSS_CONV)
        {
            attributes.reserve(36);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_W, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_H, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_C, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_N, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_K, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_S, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_R, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OUTW, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OUTH, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_STARTPADX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_STARTPADY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_ENDPADX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_ENDPADY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OUTPADX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OUTPADY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_CONVSTRIDEX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_CONVSTRIDEY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_INPUTSTRIDEX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_INPUTSTRIDEY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FILTERSTRIDEX, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FILTERSTRIDEY, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_GROUPS, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_CONV_HASBIAS, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_CONV_CROSSCORRELATION, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_CONV_BACKWARD, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DNSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DHSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DCSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FKSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FCSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FRSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FSSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_ONSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OHSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OKSTRIDE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_DOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_OOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_FOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_CONV_BOFFSET, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::CONV>(), MLSS_ATTR_CONV_DATATYPE, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSPrecisionList(), MLSS_ATTR_CONV_PRECISION, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSActivationFunctionList(), MLSS_ATTR_CONV_ACTIVATION, MLSS_ENUM));
        }
        else if (opName == MLSS_QGEMM)
        {
            attributes.reserve(29);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_QGEMM_M, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_QGEMM_N, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_QGEMM_K, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_QGEMM_BATCH, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_QGEMM_ALPHA, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_QGEMM_BETA, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASC, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_TRANSA, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_TRANSB, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASASCALE, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASAZP, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASBSCALE, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASBZP, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASCSACLE, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASCZP, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASOUTSCALE, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_QGEMM_HASOUTZP, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_QGEMM_QRATIOBSCALE, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_QGEMM_QRATIOBZP, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<std::array<std::uint32_t, 4>>(
                std::string("4 x ") + getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(),
                MLSS_ATTR_QGEMM_DATATYPEA,
                makeArrayEnum(MLSS_ARRAY | MLSS_ENUM, 4)));

            attributes.emplace_back(makeAttribute<std::array<std::uint32_t, 4>>(
                std::string("4 x ") + getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(),
                MLSS_ATTR_QGEMM_DATATYPEB,
                makeArrayEnum(MLSS_ARRAY | MLSS_ENUM, 4)));

            attributes.emplace_back(makeAttribute<std::array<std::uint32_t, 4>>(
                std::string("4 x ") + getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(),
                MLSS_ATTR_QGEMM_DATATYPEC,
                makeArrayEnum(MLSS_ARRAY | MLSS_ENUM, 4)));

            attributes.emplace_back(makeAttribute<std::array<std::uint32_t, 4>>(
                std::string("4 x ") + getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(),
                MLSS_ATTR_QGEMM_DATATYPEOUTPUT,
                makeArrayEnum(MLSS_ARRAY | MLSS_ENUM, 4)));

            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(), MLSS_ATTR_QGEMM_DATATYPEQPARAMSA, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(), MLSS_ATTR_QGEMM_DATATYPEQPARAMSB, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(), MLSS_ATTR_QGEMM_DATATYPEQPARAMSC, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::QUANTIZED_GEMM>(), MLSS_ATTR_QGEMM_DATATYPEQPARAMSOUTPUT, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSPrecisionList(), MLSS_ATTR_QGEMM_PRECISION, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSActivationFunctionList(), MLSS_ATTR_QGEMM_ACTIVATION, MLSS_ENUM));
        }
        else if (opName == MLSS_GEMM)
        {
            attributes.reserve(12);

            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GEMM_M, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GEMM_N, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GEMM_K, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<uint32_t>(uint32_t_RANGE, MLSS_ATTR_GEMM_BATCH, static_cast<std::uint32_t>(MLSS_UINT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_GEMM_ALPHA, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<float>(FLOAT32_RANGE, MLSS_ATTR_GEMM_BETA, static_cast<std::uint32_t>(MLSS_FLOAT32)));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_GEMM_HASC, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_GEMM_TRANSA, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<bool>(BOOL_RANGE, MLSS_ATTR_GEMM_TRANSB, MLSS_BOOL));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSOpSupportedTypeList<OperatorFlag::GEMM>(), MLSS_ATTR_GEMM_DATATYPE, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSPrecisionList(), MLSS_ATTR_GEMM_PRECISION, MLSS_ENUM));
            attributes.emplace_back(makeAttribute<uint32_t>(getMLSSActivationFunctionList(), MLSS_ATTR_GEMM_ACTIVATION, MLSS_ENUM));
        }
    }

} // namespace mlss
