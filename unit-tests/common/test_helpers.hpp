/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <amdmlss/amdmlss_api.h>

#include "shaders/shaders.hpp"
using ShaderDescriptor = mlss::shaders::ShaderDescriptor;

constexpr int kSkipExitCode = 77;

inline void checkStatus(MLSSstatus status, int line)
{
    if (status != MLSS_SUCCESS)
    {
        MLSSstring err = mlssGetErrorString(status);
        std::cerr << "MLSS FAIL at line " << line << ": " << err << '\n';
        std::free(err);
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
// ---------------------------------------------------------------------------

inline const MLSSbinary* findBinary(const MLSSbinary* binaries, MLSSsize count,
                                    bool wantRelocatable)
{
    for (MLSSsize i = 0; i < count; ++i)
    {
        if (static_cast<bool>(binaries[i].m_isRelocatable) == wantRelocatable)
        {
            return &binaries[i];
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Minimal fp16 <-> float conversion (IEEE 754 half-precision)
// ---------------------------------------------------------------------------

inline std::uint16_t floatToHalf(float value)
{
    std::uint32_t fbits = 0;
    std::memcpy(&fbits, &value, sizeof(fbits));

    const std::uint32_t sign = (fbits >> 16u) & 0x8000u;
    const std::int32_t exponent = static_cast<std::int32_t>((fbits >> 23u) & 0xFFu) - 127;
    const std::uint32_t mantissa = fbits & 0x007FFFFFu;

    if (exponent > 15)
    {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (exponent < -14)
    {
        return static_cast<std::uint16_t>(sign);
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent + 15) << 10u) | (mantissa >> 13u));
}

inline float halfToFloat(std::uint16_t h)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
    std::uint32_t exponent = (h >> 10u) & 0x1Fu;
    std::uint32_t mantissa = static_cast<std::uint32_t>(h & 0x03FFu) << 13u;

    if (exponent == 0x1Fu)
    {
        exponent = 0xFFu;
    }
    else if (exponent == 0u)
    {
        exponent = 0u;
        mantissa = 0u;
    }
    else
    {
        exponent += 112u;
    }

    const std::uint32_t fbits = sign | (exponent << 23u) | mantissa;
    float result = 0.0f;
    std::memcpy(&result, &fbits, sizeof(result));
    return result;
}

inline std::vector<std::uint16_t> floatsToHalves(const std::vector<float>& src)
{
    std::vector<std::uint16_t> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = floatToHalf(src[i]);
    }
    return dst;
}

inline std::vector<float> halvesToFloats(const std::vector<std::uint16_t>& src)
{
    std::vector<float> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = halfToFloat(src[i]);
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Element-wise comparison with tolerance
// ---------------------------------------------------------------------------

inline bool compareBuffers(const std::vector<float>& a, const std::vector<float>& b,
                           float tolerance, const std::string& label)
{
    if (a.size() != b.size())
    {
        std::cerr << label << ": size mismatch (" << a.size() << " vs " << b.size() << ")\n";
        return false;
    }

    bool pass = true;
    std::size_t mismatchCount = 0;
    constexpr std::size_t kMaxPrinted = 10;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > tolerance)
        {
            if (mismatchCount < kMaxPrinted)
            {
                std::cerr << label << ": mismatch at [" << i << "] "
                          << a[i] << " vs " << b[i] << " diff=" << diff << '\n';
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
// Naive host-side scaled dot-product attention (float precision)
//
// Q:      [batch, q_heads, q_seq, head_dim]
// K:      [batch, kv_heads, kv_seq, head_dim]
// V:      [batch, kv_heads, kv_seq, head_dim]
// output: [batch, q_heads, q_seq, head_dim]
//
// For MHA:  q_heads == kv_heads
// For GQA:  q_heads is a multiple of kv_heads (grouped)
// ---------------------------------------------------------------------------

inline std::vector<float> referenceAttention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::uint32_t batchSize,
    std::uint32_t qHeads,
    std::uint32_t kvHeads,
    std::uint32_t qSeq,
    std::uint32_t kvSeq,
    std::uint32_t headDim,
    float scale)
{
    const std::uint32_t headsPerGroup = qHeads / kvHeads;
    const std::size_t outputSize = static_cast<std::size_t>(batchSize) * qHeads * qSeq * headDim;
    std::vector<float> output(outputSize, 0.0f);

    for (std::uint32_t b = 0; b < batchSize; ++b)
    {
        for (std::uint32_t qh = 0; qh < qHeads; ++qh)
        {
            const std::uint32_t kvh = qh / headsPerGroup;

            for (std::uint32_t qs = 0; qs < qSeq; ++qs)
            {
                // scores[kvSeq] = Q[b,qh,qs,:] @ K[b,kvh,:,:]^T * scale
                std::vector<float> scores(kvSeq, 0.0f);
                for (std::uint32_t ks = 0; ks < kvSeq; ++ks)
                {
                    float dot = 0.0f;
                    for (std::uint32_t d = 0; d < headDim; ++d)
                    {
                        const std::size_t qIdx =
                            (static_cast<std::size_t>(b) * qHeads * qSeq + qh * qSeq + qs) * headDim + d;
                        const std::size_t kIdx =
                            (static_cast<std::size_t>(b) * kvHeads * kvSeq + kvh * kvSeq + ks) * headDim + d;
                        dot += Q[qIdx] * K[kIdx];
                    }
                    scores[ks] = dot * scale;
                }

                // softmax
                float maxScore = *std::max_element(scores.begin(), scores.end());
                float sumExp = 0.0f;
                for (auto& s : scores)
                {
                    s = std::exp(s - maxScore);
                    sumExp += s;
                }
                for (auto& s : scores)
                {
                    s /= sumExp;
                }

                // output[b,qh,qs,:] = scores @ V[b,kvh,:,:]
                for (std::uint32_t d = 0; d < headDim; ++d)
                {
                    float val = 0.0f;
                    for (std::uint32_t ks = 0; ks < kvSeq; ++ks)
                    {
                        const std::size_t vIdx =
                            (static_cast<std::size_t>(b) * kvHeads * kvSeq + kvh * kvSeq + ks) * headDim + d;
                        val += scores[ks] * V[vIdx];
                    }
                    const std::size_t oIdx =
                        (static_cast<std::size_t>(b) * qHeads * qSeq + qh * qSeq + qs) * headDim + d;
                    output[oIdx] = val;
                }
            }
        }
    }

    return output;
}

// ---------------------------------------------------------------------------
// Random fp16-representable float generator (small magnitude for stability)
// ---------------------------------------------------------------------------

inline std::vector<float> generateRandomFloats(std::size_t count, float lo, float hi,
                                               std::uint32_t seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(count);
    for (auto& x : v)
    {
        x = halfToFloat(floatToHalf(dist(rng)));
    }
    return v;
}
