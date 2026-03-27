/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::shaders::mha::ck::wmma::fp16
{
    // Shader constants for different MHA configurations
    constexpr auto cross_attention_128_64x64x48_64x48x64_forward_CONSTANTS = std::to_array<std::uint32_t>({64, 48, 128, 1, 1});
    constexpr auto fallback_cross_attention_64_32x64x48_32x48x64_forward_CONSTANTS = std::to_array<std::uint32_t>({32, 48, 64, 1, 1});
    constexpr auto self_attention_128_64x128x80_64x80x64_forward_CONSTANTS = std::to_array<std::uint32_t>({64, 80, 128, 1, 1});
    constexpr auto self_attention_128_64x192x48_64x48x64_forward_CONSTANTS = std::to_array<std::uint32_t>({64, 48, 128, 1, 1});
    constexpr auto fallback_self_attention_64_32x64x48_32x48x64_forward_CONSTANTS = std::to_array<std::uint32_t>({32, 48, 64, 1, 1});

    // Argument definitions for packed Q-KV cross attention
    const std::array<MLSSarg, 9> packed_q_kv_cross_attention_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                                {1, MLSS_FLOAT16, true, 2, true, true, false, "KV"},
                                                                                {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                {6, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                                {7, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                {8, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QKV self attention
    const std::array<MLSSarg, 7> packed_qkv_self_attention_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QKV"},
                                                                              {1, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                              {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {6, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for unpacked Q-K-V cross attention
    const std::array<MLSSarg, 10> unpacked_q_k_v_cross_attention_ARGS_CONSTANT = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                                   {1, MLSS_FLOAT16, true, 2, true, true, false, "K"},
                                                                                   {2, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                                   {3, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                   {7, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                                   {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for unpacked Q-K-V self attention
    const std::array<MLSSarg, 9> unpacked_q_k_v_self_attention_ARGS_CONSTANT = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                                 {1, MLSS_FLOAT16, true, 2, true, true, false, "K"},
                                                                                 {2, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                                 {3, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                 {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                 {5, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                 {6, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                                 {7, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                 {8, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QK MHA (Q and K packed, V separate)
    const std::array<MLSSarg, 9> packed_qk_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QK"}, // Packed Q and K
                                                              {1, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                              {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                              {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                              {6, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                              {7, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                              {8, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QK MHA with strides
    const std::array<MLSSarg, 21> packed_qk_with_strides_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QK"}, // Packed Q and K
                                                                            {1, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                            {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                            {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                            {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                            {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                            {6, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                            {7, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                            {8, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                            {9, MLSS_INT32, false, 0, true, true, false, "qk_stride_d0"},
                                                                            {10, MLSS_INT32, false, 0, true, true, false, "qk_stride_d1"},
                                                                            {11, MLSS_INT32, false, 0, true, true, false, "qk_stride_d2"},
                                                                            {12, MLSS_INT32, false, 0, true, true, false, "qk_stride_d3"},
                                                                            {13, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                            {14, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                            {15, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                            {16, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                            {17, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                            {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                            {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                            {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for unpacked Q-K-V with strides
    const std::array<MLSSarg, 26> unpacked_q_k_v_with_strides_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                                 {1, MLSS_FLOAT16, true, 2, true, true, false, "K"},
                                                                                 {2, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                                 {3, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                 {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                 {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                 {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                 {7, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                                 {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                 {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                 {10, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                 {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                 {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                 {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                 {14, MLSS_INT32, false, 0, true, true, false, "k_stride_d0"},
                                                                                 {15, MLSS_INT32, false, 0, true, true, false, "k_stride_d1"},
                                                                                 {16, MLSS_INT32, false, 0, true, true, false, "k_stride_d2"},
                                                                                 {17, MLSS_INT32, false, 0, true, true, false, "k_stride_d3"},
                                                                                 {18, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                 {19, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                 {20, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                 {21, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                 {22, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                 {23, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                 {24, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                 {25, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed Q-KV with strides
    const std::array<MLSSarg, 21> packed_q_kv_with_strides_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                              {1, MLSS_FLOAT16, true, 2, true, true, false, "KV"}, // Packed K and V
                                                                              {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                              {7, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {8, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                              {9, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                              {10, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                              {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                              {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                              {13, MLSS_INT32, false, 0, true, true, false, "kv_stride_d0"},
                                                                              {14, MLSS_INT32, false, 0, true, true, false, "kv_stride_d1"},
                                                                              {15, MLSS_INT32, false, 0, true, true, false, "kv_stride_d2"},
                                                                              {16, MLSS_INT32, false, 0, true, true, false, "kv_stride_d3"},
                                                                              {17, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                              {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                              {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                              {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QKV with strides
    const std::array<MLSSarg, 15> packed_qkv_with_strides_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QKV"}, // Packed Q, K and V
                                                                             {1, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                             {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                             {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                             {4, MLSS_INT32, false, 0, true, true, false, "head_num"},
                                                                             {5, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                             {6, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                             {7, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d0"},
                                                                             {8, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d1"},
                                                                             {9, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d2"},
                                                                             {10, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d3"},
                                                                             {11, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                             {12, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                             {13, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                             {14, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

} // namespace mlss::shaders::mha::ck::wmma::fp16
