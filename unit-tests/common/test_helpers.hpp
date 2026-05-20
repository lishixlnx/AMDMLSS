/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <amdmlss/amdmlss_api.h>

#include "shaders/shaders.hpp"
using ShaderDescriptor = mlss::shaders::ShaderDescriptor;

#include "common/kernelArg.hpp"
#include "host/tensor_host.h"

constexpr int kSkipExitCode = 77;

inline void checkStatus(MLSSstatus status, int line)
{
    if (status != MLSS_SUCCESS)
    {
        MLSSstring err = mlssGetErrorString(status);
        std::cerr << "MLSS FAIL at line " << line << ": " << err << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_STATUS(status) checkStatus((status), __LINE__)

// ---------------------------------------------------------------------------
// ShaderDescriptor builder from MLSSbinary
// ---------------------------------------------------------------------------

inline ShaderDescriptor buildShaderDescriptor(const MLSSbinary& bin)
{
    ShaderDescriptor desc;
    const auto* raw = static_cast<const std::byte*>(bin.m_binaries);
    desc.m_binary.assign(raw, raw + bin.m_binarySize);
    desc.m_kernelName = bin.m_pKernelName ? bin.m_pKernelName : "";
    return desc;
}

// ---------------------------------------------------------------------------
// Binary finder: returns pointer to the first MLSSbinary with the requested
// relocatable flag, or nullptr if none found.
// When preferSmallArgList is true, returns the match with the fewest
// m_argList entries (i.e. the "no strides" variant).
// ---------------------------------------------------------------------------

inline const MLSSbinary* findBinary(const MLSSbinary* binaries, MLSSsize count,
                                    bool wantRelocatable,
                                    bool preferSmallArgList = false)
{
    const MLSSbinary* best = nullptr;

    for (const auto* p = binaries; p < binaries + count; ++p)
    {
        if (static_cast<bool>(p->m_isRelocatable) != wantRelocatable)
            continue;

        if (best == nullptr)
        {
            best = p;
        }
        else if (preferSmallArgList)
        {
            if (p->m_argList.m_size < best->m_argList.m_size)
                best = p;
        }
        else
        {
            if (p->m_argList.m_size > best->m_argList.m_size)
                best = p;
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Build kernel arguments from MLSSbinary.m_argList
//
// Reads the self-describing argument list via mlssVectorRetrieveData, then
// assembles a vector<KernelArg> in m_place order by looking up each argument
// name in the caller-provided map.
// ---------------------------------------------------------------------------

inline std::vector<KernelArg> buildArgsFromBinary(
    const MLSSbinary& bin,
    const std::unordered_map<std::string, KernelArg>& argMap)
{
    MLSSvoid* rawData  = nullptr;
    MLSSsize  argCount = 0;
    MLSSenum  argType  = 0;

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
        auto it = argMap.find(name);
        if (it == argMap.end())
        {
            std::cerr << "buildArgsFromBinary: no value for argument '"
                      << name << "' (place=" << arg.m_place << ")\n";
            return {};
        }
        args.push_back(it->second);
    }

    return args;
}

// ---------------------------------------------------------------------------
// fp16 type alias — prefer std::float16_t (C++23), fall back to _Float16
// ---------------------------------------------------------------------------

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
using float16_t = std::float16_t;
#elif defined(__FLT16_MAX__)
using float16_t = _Float16;
#else
struct float16_storage
{
    std::uint16_t bits = 0;
};
using float16_t = float16_storage;

#endif

static_assert(sizeof(float16_t) == 2);

// ---------------------------------------------------------------------------
// fp16 <-> float conversion
// ---------------------------------------------------------------------------

#if defined(__STDCPP_FLOAT16_T__)

inline float16_t floatToHalf(float value) { return static_cast<float16_t>(value); }
inline float halfToFloat(float16_t h)     { return static_cast<float>(h); }

#else

inline float16_t floatToHalf(float value)
{
    std::uint32_t fbits;
    std::memcpy(&fbits, &value, sizeof(float));

    const std::uint32_t sign = (fbits >> 16) & 0x8000u;
    const std::int32_t  exp  = static_cast<std::int32_t>((fbits >> 23) & 0xFFu) - 127 + 15;
    const std::uint32_t frac = (fbits >> 13) & 0x03FFu;

    std::uint16_t half;
    if (exp <= 0)
        half = static_cast<std::uint16_t>(sign);
    else if (exp >= 31)
        half = static_cast<std::uint16_t>(sign | 0x7C00u);
    else
        half = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | frac);

    float16_t result;
    std::memcpy(&result, &half, sizeof(float16_t));
    return result;
}

inline float halfToFloat(float16_t h)
{
    std::uint16_t hbits;
    std::memcpy(&hbits, &h, sizeof(std::uint16_t));

    const std::uint32_t sign = static_cast<std::uint32_t>(hbits & 0x8000u) << 16;
    const std::uint32_t exp  = (hbits >> 10) & 0x1Fu;
    const std::uint32_t frac = hbits & 0x03FFu;

    std::uint32_t fbits;
    if (exp == 0)
        fbits = sign;
    else if (exp == 31)
        fbits = sign | 0x7F800000u | (frac << 13);
    else
        fbits = sign | ((exp - 15 + 127) << 23) | (frac << 13);

    float result;
    std::memcpy(&result, &fbits, sizeof(float));
    return result;
}

#endif

// ---------------------------------------------------------------------------
// TensorHost <-> fp16 vector conversion (GPU upload / download)
// ---------------------------------------------------------------------------

inline std::vector<float16_t> tensorToHalves(const TensorHost<float>& t)
{
    std::vector<float16_t> out(t.numel());
    const float* src = t.data();
    for (std::size_t i = 0; i < t.numel(); ++i)
        out[i] = floatToHalf(src[i]);
    return out;
}

inline TensorHost<float> halvesToTensor(const std::vector<float16_t>& halves,
                                        std::vector<std::uint32_t> shape)
{
    TensorHost<float> t(std::move(shape));
    assert(t.numel() == halves.size());
    for (std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = halfToFloat(halves[i]);
    return t;
}

inline TensorHost<float> vectorToTensor(const std::vector<float>& src,
                                        std::vector<std::uint32_t> shape)
{
    TensorHost<float> t(std::move(shape));
    assert(t.numel() == src.size());
    std::memcpy(t.data(), src.data(), src.size() * sizeof(float));
    return t;
}

inline std::vector<float> halvesToFloats(const std::vector<float16_t>& src)
{
    std::vector<float> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst[i] = halfToFloat(src[i]);
    return dst;
}

// ---------------------------------------------------------------------------
// Convert TensorHost<float> to flat std::vector<float>
inline std::vector<float> tensorToVector(const TensorHost<float>& tensor)
{
    const float* data = tensor.data();
    return std::vector<float>(data, data + tensor.numel());
}

inline bool convCompareBuffers(const std::vector<float>& a, const TensorHost<float>& b)
{
/*    for (auto i : a)
        std::cout << i << std::endl;*/
    auto B = tensorToVector(b);
    float diff = 0;
    for (auto i = 0; i < a.size(); ++i)
    {
        diff += B[i] - a[i];
    }
    std::cout << "Difference = " << diff << std::endl;
    return false;
}

// ---------------------------------------------------------------------------
// Element-wise comparison with tolerance
// ---------------------------------------------------------------------------
inline bool compareBuffers(const TensorHost<float>& a, const TensorHost<float>& b,
                           float tolerance, const std::string& label)
{
    if (a.numel() != b.numel())
    {
        std::cerr << label << ": size mismatch (" << a.numel() << " vs " << b.numel() << ")\n";
        return false;
    }

    bool pass = true;
    std::size_t mismatchCount = 0;
    constexpr std::size_t kMaxPrinted = 10;

    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        const float diff = std::fabs(a.data()[i] - b.data()[i]);
        if (diff > tolerance)
        {
            if (mismatchCount < kMaxPrinted)
            {
                std::cerr << label << ": mismatch at [" << i << "] "
                          << a.data()[i] << " vs " << b.data()[i] << " diff=" << diff << '\n';
            }
            ++mismatchCount;
            pass = false;
        }
    }

    if (mismatchCount > kMaxPrinted)
    {
        std::cerr << label << ": ... and " << (mismatchCount - kMaxPrinted)
                  << " more mismatches\n";
    }

    return pass;
}

// ---------------------------------------------------------------------------
// Host-side scaled dot-product attention using TensorHost + BLAS
//
// Q:      [batch, q_heads, q_seq, head_dim]
// K:      [batch, kv_heads, kv_seq, head_dim]
// V:      [batch, kv_heads, kv_seq, head_dim]
// output: [batch, q_heads, q_seq, head_dim]
//
// For MHA:  q_heads == kv_heads
// For GQA:  q_heads is a multiple of kv_heads (grouped)
// ---------------------------------------------------------------------------

inline TensorHost<float> referenceAttention(
    const TensorHost<float>& Q,
    const TensorHost<float>& K,
    const TensorHost<float>& V,
    float scale)
{
    const auto& qShape = Q.shape();
    const std::uint32_t batchSize = qShape[0];
    const std::uint32_t qHeads   = qShape[1];
    const std::uint32_t qSeq     = qShape[2];
    const std::uint32_t headDim  = qShape[3];

    const std::uint32_t kvHeads = K.shape()[1];
    const std::uint32_t kvSeq   = K.shape()[2];
    const std::uint32_t headsPerGroup = qHeads / kvHeads;

    TensorHost<float> output(batchSize, qHeads, qSeq, headDim);

    for (std::uint32_t b = 0; b < batchSize; ++b)
    {
        for (std::uint32_t qh = 0; qh < qHeads; ++qh)
        {
            const std::uint32_t kvh = qh / headsPerGroup;

            // scores = Q[b,qh] @ K[b,kvh]^T * scale   →  [qSeq x kvSeq]
            TensorHost<float> scores(qSeq, kvSeq);

            th_blas::gemm<float>(
                false, true,
                qSeq, kvSeq, headDim,
                scale,  &Q(b, qh, 0u, 0u), headDim,
                        &K(b, kvh, 0u, 0u), headDim,
                0.0f,   scores.data(), kvSeq);

            // row-wise softmax
            for (std::uint32_t qs = 0; qs < qSeq; ++qs)
            {
                float maxVal = scores(qs, 0u);
                for (std::uint32_t ks = 1; ks < kvSeq; ++ks)
                    maxVal = std::max(maxVal, scores(qs, ks));

                float sumExp = 0.0f;
                for (std::uint32_t ks = 0; ks < kvSeq; ++ks)
                {
                    scores(qs, ks) = std::exp(scores(qs, ks) - maxVal);
                    sumExp += scores(qs, ks);
                }
                for (std::uint32_t ks = 0; ks < kvSeq; ++ks)
                    scores(qs, ks) /= sumExp;
            }

            // output[b,qh] = scores @ V[b,kvh]   →  [qSeq x headDim]
            th_blas::gemm<float>(
                false, false,
                qSeq, headDim, kvSeq,
                1.0f,  scores.data(), kvSeq,
                       &V(b, kvh, 0u, 0u), headDim,
                0.0f,  &output(b, qh, 0u, 0u), headDim);
        }
    }

    return output;
}

// ---------------------------------------------------------------------------
// Host-side conv computation 
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 4D tensor axis-1/axis-2 transpose: [B, X, Y, D] ↔ [B, Y, X, D]
//
// Used to convert between BHSD (host reference) and BSHD (GPU kernel layout).
// The operation is self-inverse: applying it twice yields the original tensor.
// ---------------------------------------------------------------------------

inline TensorHost<float> transposeHeadSeq(const TensorHost<float>& t)
{
    const auto& sh = t.shape();
    const std::uint32_t d0 = sh[0];
    const std::uint32_t d1 = sh[1];
    const std::uint32_t d2 = sh[2];
    const std::uint32_t d3 = sh[3];

    TensorHost<float> out(d0, d2, d1, d3);
    for (std::uint32_t i0 = 0; i0 < d0; ++i0)
        for (std::uint32_t i1 = 0; i1 < d1; ++i1)
            for (std::uint32_t i2 = 0; i2 < d2; ++i2)
                for (std::uint32_t i3 = 0; i3 < d3; ++i3)
                    out(i0, i2, i1, i3) = t(i0, i1, i2, i3);
    return out;
}

// ---------------------------------------------------------------------------
// Random fp16-representable TensorHost generator (small magnitude for stability)
// ---------------------------------------------------------------------------

inline TensorHost<float> generateRandomTensor(std::vector<std::uint32_t> shape,
                                              float lo, float hi,
                                              std::uint32_t seed = 42)
{
    TensorHost<float> t(std::move(shape));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = halfToFloat(floatToHalf(dist(rng)));
    return t;
}

// index_t, GQAPackingFlags, has_flag and calcStrides are provided by the
// mlss-tester library (common/misc.hpp, pulled in via <mlss_tester.hpp>).
#include "common/misc.hpp"