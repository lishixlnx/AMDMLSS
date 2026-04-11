/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

namespace mlss::gqa::ck::wmma::fp16
{
    // Shader constants for different GQA configurations
    // Format: { MPerBlock, NPerBlock, BlockSize, ... }
    constexpr auto gqa_128_64x128x80_64x80x64_CONSTANTS = std::to_array<std::uint32_t>({64, 80, 128, 1, 1});
    constexpr auto gqa_128_64x192x48_64x48x64_CONSTANTS = std::to_array<std::uint32_t>({64, 48, 128, 1, 1});
    constexpr auto gqa_128_64x64x48_64x48x64_CONSTANTS = std::to_array<std::uint32_t>({64, 48, 128, 1, 1});
    constexpr auto gqa_fallback_64_32x64x48_32x48x64_CONSTANTS = std::to_array<std::uint32_t>({32, 48, 64, 1, 1});

    // Argument definitions for unpacked GQA (double pointer variant)
    const std::array<MLSSarg, 11> unpacked_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                             {1, MLSS_FLOAT16, true, 2, true, true, false, "K"},
                                                                             {2, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                             {3, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                             {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                             {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                             {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                             {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                             {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                             {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                             {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for unpacked GQA with strides (double pointer variant)
    const std::array<MLSSarg, 27> unpacked_with_strides_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                                          {1, MLSS_FLOAT16, true, 2, true, true, false, "K"},
                                                                                          {2, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                                          {3, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                          {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                          {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                          {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                          {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                          {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                          {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                          {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                          {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                          {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                          {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                          {14, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                          {15, MLSS_INT32, false, 0, true, true, false, "k_stride_d0"},
                                                                                          {16, MLSS_INT32, false, 0, true, true, false, "k_stride_d1"},
                                                                                          {17, MLSS_INT32, false, 0, true, true, false, "k_stride_d2"},
                                                                                          {18, MLSS_INT32, false, 0, true, true, false, "k_stride_d3"},
                                                                                          {19, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                          {20, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                          {21, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                          {22, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                          {23, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                          {24, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                          {25, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                          {26, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QK GQA (double pointer variant)
    // Q and K are packed together, V is separate
    const std::array<MLSSarg, 10> packed_qk_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QK"}, // Packed Q and K
                                                                              {1, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                              {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                              {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                              {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QK GQA with strides (double pointer variant)
    const std::array<MLSSarg, 22> packed_qk_with_strides_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QK"}, // Packed Q and K
                                                                                           {1, MLSS_FLOAT16, true, 2, true, true, false, "V"},
                                                                                           {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                           {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                           {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                           {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                           {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                           {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                           {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                           {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                           {10, MLSS_INT32, false, 0, true, true, false, "qk_stride_d0"},
                                                                                           {11, MLSS_INT32, false, 0, true, true, false, "qk_stride_d1"},
                                                                                           {12, MLSS_INT32, false, 0, true, true, false, "qk_stride_d2"},
                                                                                           {13, MLSS_INT32, false, 0, true, true, false, "qk_stride_d3"},
                                                                                           {14, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                           {15, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                           {16, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                           {17, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                           {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                           {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                           {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                           {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed KV GQA (double pointer variant)
    // Q is separate, K and V are packed together
    const std::array<MLSSarg, 10> packed_kv_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                              {1, MLSS_FLOAT16, true, 2, true, true, false, "KV"}, // Packed K and V
                                                                              {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                              {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                              {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed KV GQA with strides (double pointer variant)
    const std::array<MLSSarg, 22> packed_kv_with_strides_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "Q"},
                                                                                           {1, MLSS_FLOAT16, true, 2, true, true, false, "KV"}, // Packed K and V
                                                                                           {2, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                           {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                           {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                           {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                           {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                           {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                           {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                           {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                           {10, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                           {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                           {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                           {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                           {14, MLSS_INT32, false, 0, true, true, false, "kv_stride_d0"},
                                                                                           {15, MLSS_INT32, false, 0, true, true, false, "kv_stride_d1"},
                                                                                           {16, MLSS_INT32, false, 0, true, true, false, "kv_stride_d2"},
                                                                                           {17, MLSS_INT32, false, 0, true, true, false, "kv_stride_d3"},
                                                                                           {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                           {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                           {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                           {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QKV GQA (double pointer variant)
    // Q, K, and V are all packed together
    const std::array<MLSSarg, 8> packed_qkv_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QKV"}, // Packed Q, K, and V
                                                                              {1, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                              {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QKV GQA with strides (double pointer variant)
    const std::array<MLSSarg, 16> packed_qkv_with_strides_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 2, true, true, false, "QKV"}, // Packed Q, K, and V
                                                                                            {1, MLSS_FLOAT16, true, 2, false, false, true, "output"},
                                                                                            {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                            {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                            {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                            {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                            {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                            {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                            {8, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d0"},
                                                                                            {9, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d1"},
                                                                                            {10, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d2"},
                                                                                            {11, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d3"},
                                                                                            {12, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                            {13, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                            {14, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                            {15, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    //=========================================================================
    // SINGLE POINTER VARIANTS (indirection level = 1)
    //=========================================================================

    // Argument definitions for unpacked GQA (single pointer variant)
    const std::array<MLSSarg, 11> unpacked_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "Q"},
                                                                             {1, MLSS_FLOAT16, true, 1, true, true, false, "K"},
                                                                             {2, MLSS_FLOAT16, true, 1, true, true, false, "V"},
                                                                             {3, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                             {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                             {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                             {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                             {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                             {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                             {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                             {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for unpacked GQA with strides (single pointer variant)
    const std::array<MLSSarg, 27> unpacked_with_strides_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "Q"},
                                                                                          {1, MLSS_FLOAT16, true, 1, true, true, false, "K"},
                                                                                          {2, MLSS_FLOAT16, true, 1, true, true, false, "V"},
                                                                                          {3, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                                          {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                          {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                          {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                          {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                          {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                          {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                          {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                          {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                          {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                          {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                          {14, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                          {15, MLSS_INT32, false, 0, true, true, false, "k_stride_d0"},
                                                                                          {16, MLSS_INT32, false, 0, true, true, false, "k_stride_d1"},
                                                                                          {17, MLSS_INT32, false, 0, true, true, false, "k_stride_d2"},
                                                                                          {18, MLSS_INT32, false, 0, true, true, false, "k_stride_d3"},
                                                                                          {19, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                          {20, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                          {21, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                          {22, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                          {23, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                          {24, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                          {25, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                          {26, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QK GQA (single pointer variant)
    const std::array<MLSSarg, 10> packed_qk_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "QK"},
                                                                              {1, MLSS_FLOAT16, true, 1, true, true, false, "V"},
                                                                              {2, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                              {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                              {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QK GQA with strides (single pointer variant)
    const std::array<MLSSarg, 22> packed_qk_with_strides_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "QK"},
                                                                                           {1, MLSS_FLOAT16, true, 1, true, true, false, "V"},
                                                                                           {2, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                                           {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                           {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                           {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                           {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                           {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                           {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                           {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                           {10, MLSS_INT32, false, 0, true, true, false, "qk_stride_d0"},
                                                                                           {11, MLSS_INT32, false, 0, true, true, false, "qk_stride_d1"},
                                                                                           {12, MLSS_INT32, false, 0, true, true, false, "qk_stride_d2"},
                                                                                           {13, MLSS_INT32, false, 0, true, true, false, "qk_stride_d3"},
                                                                                           {14, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                           {15, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                           {16, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                           {17, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                           {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                           {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                           {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                           {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed KV GQA (single pointer variant)
    const std::array<MLSSarg, 10> packed_kv_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "Q"},
                                                                              {1, MLSS_FLOAT16, true, 1, true, true, false, "KV"},
                                                                              {2, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                              {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                              {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed KV GQA with strides (single pointer variant)
    const std::array<MLSSarg, 22> packed_kv_with_strides_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "Q"},
                                                                                           {1, MLSS_FLOAT16, true, 1, true, true, false, "KV"},
                                                                                           {2, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                                           {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                           {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                           {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                           {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                           {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                           {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                           {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                           {10, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                           {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                           {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                           {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                           {14, MLSS_INT32, false, 0, true, true, false, "kv_stride_d0"},
                                                                                           {15, MLSS_INT32, false, 0, true, true, false, "kv_stride_d1"},
                                                                                           {16, MLSS_INT32, false, 0, true, true, false, "kv_stride_d2"},
                                                                                           {17, MLSS_INT32, false, 0, true, true, false, "kv_stride_d3"},
                                                                                           {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                           {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                           {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                           {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QKV GQA (single pointer variant)
    const std::array<MLSSarg, 8> packed_qkv_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "QKV"},
                                                                              {1, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                              {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                              {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                              {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                              {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                              {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                              {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QKV GQA with strides (single pointer variant)
    const std::array<MLSSarg, 16> packed_qkv_with_strides_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_FLOAT16, true, 1, true, true, false, "QKV"},
                                                                                            {1, MLSS_FLOAT16, true, 1, false, false, true, "output"},
                                                                                            {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                            {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                            {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                            {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                            {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                            {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                            {8, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d0"},
                                                                                            {9, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d1"},
                                                                                            {10, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d2"},
                                                                                            {11, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d3"},
                                                                                            {12, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                            {13, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                            {14, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                            {15, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    //=========================================================================
    // VOID DOUBLE POINTER VARIANTS (indirection level = 2, MLSS_VOID type)
    //=========================================================================

    // Argument definitions for unpacked GQA (void double pointer variant)
    const std::array<MLSSarg, 11> unpacked_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "Q"},
                                                                                  {1, MLSS_VOID, true, 2, true, true, false, "K"},
                                                                                  {2, MLSS_VOID, true, 2, true, true, false, "V"},
                                                                                  {3, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                  {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                  {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                  {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                  {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                  {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                  {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                  {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for unpacked GQA with strides (void double pointer variant)
    const std::array<MLSSarg, 27> unpacked_with_strides_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "Q"},
                                                                                               {1, MLSS_VOID, true, 2, true, true, false, "K"},
                                                                                               {2, MLSS_VOID, true, 2, true, true, false, "V"},
                                                                                               {3, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                               {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                               {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                               {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                               {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                               {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                               {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                               {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                               {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                               {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                               {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                               {14, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                               {15, MLSS_INT32, false, 0, true, true, false, "k_stride_d0"},
                                                                                               {16, MLSS_INT32, false, 0, true, true, false, "k_stride_d1"},
                                                                                               {17, MLSS_INT32, false, 0, true, true, false, "k_stride_d2"},
                                                                                               {18, MLSS_INT32, false, 0, true, true, false, "k_stride_d3"},
                                                                                               {19, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                               {20, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                               {21, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                               {22, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                               {23, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                               {24, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                               {25, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                               {26, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QK GQA (void double pointer variant)
    const std::array<MLSSarg, 10> packed_qk_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "QK"},
                                                                                   {1, MLSS_VOID, true, 2, true, true, false, "V"},
                                                                                   {2, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                   {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                   {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                   {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QK GQA with strides (void double pointer variant)
    const std::array<MLSSarg, 22> packed_qk_with_strides_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "QK"},
                                                                                                {1, MLSS_VOID, true, 2, true, true, false, "V"},
                                                                                                {2, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                                {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                                {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                                {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                                {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                                {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                                {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                                {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                                {10, MLSS_INT32, false, 0, true, true, false, "qk_stride_d0"},
                                                                                                {11, MLSS_INT32, false, 0, true, true, false, "qk_stride_d1"},
                                                                                                {12, MLSS_INT32, false, 0, true, true, false, "qk_stride_d2"},
                                                                                                {13, MLSS_INT32, false, 0, true, true, false, "qk_stride_d3"},
                                                                                                {14, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                                {15, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                                {16, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                                {17, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                                {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                                {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                                {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                                {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed KV GQA (void double pointer variant)
    const std::array<MLSSarg, 10> packed_kv_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "Q"},
                                                                                   {1, MLSS_VOID, true, 2, true, true, false, "KV"},
                                                                                   {2, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                   {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                   {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                   {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed KV GQA with strides (void double pointer variant)
    const std::array<MLSSarg, 22> packed_kv_with_strides_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "Q"},
                                                                                                {1, MLSS_VOID, true, 2, true, true, false, "KV"},
                                                                                                {2, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                                {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                                {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                                {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                                {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                                {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                                {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                                {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                                {10, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                                {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                                {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                                {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                                {14, MLSS_INT32, false, 0, true, true, false, "kv_stride_d0"},
                                                                                                {15, MLSS_INT32, false, 0, true, true, false, "kv_stride_d1"},
                                                                                                {16, MLSS_INT32, false, 0, true, true, false, "kv_stride_d2"},
                                                                                                {17, MLSS_INT32, false, 0, true, true, false, "kv_stride_d3"},
                                                                                                {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                                {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                                {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                                {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QKV GQA (void double pointer variant)
    const std::array<MLSSarg, 8> packed_qkv_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "QKV"},
                                                                                   {1, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                   {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QKV GQA with strides (void double pointer variant)
    const std::array<MLSSarg, 16> packed_qkv_with_strides_void_double_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 2, true, true, false, "QKV"},
                                                                                                 {1, MLSS_VOID, true, 2, false, false, true, "output"},
                                                                                                 {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                                 {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                                 {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                                 {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                                 {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                                 {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                                 {8, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d0"},
                                                                                                 {9, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d1"},
                                                                                                 {10, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d2"},
                                                                                                 {11, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d3"},
                                                                                                 {12, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                                 {13, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                                 {14, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                                 {15, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    //=========================================================================
    // VOID SINGLE POINTER VARIANTS (indirection level = 1, MLSS_VOID type)
    //=========================================================================

    // Argument definitions for unpacked GQA (void single pointer variant)
    const std::array<MLSSarg, 11> unpacked_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "Q"},
                                                                                  {1, MLSS_VOID, true, 1, true, true, false, "K"},
                                                                                  {2, MLSS_VOID, true, 1, true, true, false, "V"},
                                                                                  {3, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                  {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                  {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                  {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                  {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                  {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                  {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                  {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for unpacked GQA with strides (void single pointer variant)
    const std::array<MLSSarg, 27> unpacked_with_strides_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "Q"},
                                                                                               {1, MLSS_VOID, true, 1, true, true, false, "K"},
                                                                                               {2, MLSS_VOID, true, 1, true, true, false, "V"},
                                                                                               {3, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                               {4, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                               {5, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                               {6, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                               {7, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                               {8, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                               {9, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                               {10, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                               {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                               {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                               {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                               {14, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                               {15, MLSS_INT32, false, 0, true, true, false, "k_stride_d0"},
                                                                                               {16, MLSS_INT32, false, 0, true, true, false, "k_stride_d1"},
                                                                                               {17, MLSS_INT32, false, 0, true, true, false, "k_stride_d2"},
                                                                                               {18, MLSS_INT32, false, 0, true, true, false, "k_stride_d3"},
                                                                                               {19, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                               {20, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                               {21, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                               {22, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                               {23, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                               {24, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                               {25, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                               {26, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QK GQA (void single pointer variant)
    const std::array<MLSSarg, 10> packed_qk_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "QK"},
                                                                                   {1, MLSS_VOID, true, 1, true, true, false, "V"},
                                                                                   {2, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                   {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                   {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                   {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QK GQA with strides (void single pointer variant)
    const std::array<MLSSarg, 22> packed_qk_with_strides_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "QK"},
                                                                                                {1, MLSS_VOID, true, 1, true, true, false, "V"},
                                                                                                {2, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                                {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                                {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                                {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                                {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                                {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                                {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                                {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                                {10, MLSS_INT32, false, 0, true, true, false, "qk_stride_d0"},
                                                                                                {11, MLSS_INT32, false, 0, true, true, false, "qk_stride_d1"},
                                                                                                {12, MLSS_INT32, false, 0, true, true, false, "qk_stride_d2"},
                                                                                                {13, MLSS_INT32, false, 0, true, true, false, "qk_stride_d3"},
                                                                                                {14, MLSS_INT32, false, 0, true, true, false, "v_stride_d0"},
                                                                                                {15, MLSS_INT32, false, 0, true, true, false, "v_stride_d1"},
                                                                                                {16, MLSS_INT32, false, 0, true, true, false, "v_stride_d2"},
                                                                                                {17, MLSS_INT32, false, 0, true, true, false, "v_stride_d3"},
                                                                                                {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                                {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                                {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                                {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed KV GQA (void single pointer variant)
    const std::array<MLSSarg, 10> packed_kv_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "Q"},
                                                                                   {1, MLSS_VOID, true, 1, true, true, false, "KV"},
                                                                                   {2, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                   {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                   {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                   {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed KV GQA with strides (void single pointer variant)
    const std::array<MLSSarg, 22> packed_kv_with_strides_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "Q"},
                                                                                                {1, MLSS_VOID, true, 1, true, true, false, "KV"},
                                                                                                {2, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                                {3, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                                {4, MLSS_INT32, false, 0, true, true, false, "q_sequence_length"},
                                                                                                {5, MLSS_INT32, false, 0, true, true, false, "kv_sequence_length"},
                                                                                                {6, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                                {7, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                                {8, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                                {9, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                                {10, MLSS_INT32, false, 0, true, true, false, "q_stride_d0"},
                                                                                                {11, MLSS_INT32, false, 0, true, true, false, "q_stride_d1"},
                                                                                                {12, MLSS_INT32, false, 0, true, true, false, "q_stride_d2"},
                                                                                                {13, MLSS_INT32, false, 0, true, true, false, "q_stride_d3"},
                                                                                                {14, MLSS_INT32, false, 0, true, true, false, "kv_stride_d0"},
                                                                                                {15, MLSS_INT32, false, 0, true, true, false, "kv_stride_d1"},
                                                                                                {16, MLSS_INT32, false, 0, true, true, false, "kv_stride_d2"},
                                                                                                {17, MLSS_INT32, false, 0, true, true, false, "kv_stride_d3"},
                                                                                                {18, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                                {19, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                                {20, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                                {21, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

    // Argument definitions for packed QKV GQA (void single pointer variant)
    const std::array<MLSSarg, 8> packed_qkv_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "QKV"},
                                                                                   {1, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                   {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                   {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                   {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                   {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                   {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                   {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"}}};

    // Argument definitions for packed QKV GQA with strides (void single pointer variant)
    const std::array<MLSSarg, 16> packed_qkv_with_strides_void_single_pointer_ARGS_CONSTANTS = {{{0, MLSS_VOID, true, 1, true, true, false, "QKV"},
                                                                                                 {1, MLSS_VOID, true, 1, false, false, true, "output"},
                                                                                                 {2, MLSS_INT32, false, 0, true, true, false, "batch_size"},
                                                                                                 {3, MLSS_INT32, false, 0, true, true, false, "sequence_length"},
                                                                                                 {4, MLSS_INT32, false, 0, true, true, false, "q_head_num"},
                                                                                                 {5, MLSS_INT32, false, 0, true, true, false, "kv_head_num"},
                                                                                                 {6, MLSS_INT32, false, 0, true, true, false, "head_dim"},
                                                                                                 {7, MLSS_FLOAT32, false, 0, true, true, false, "scale"},
                                                                                                 {8, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d0"},
                                                                                                 {9, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d1"},
                                                                                                 {10, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d2"},
                                                                                                 {11, MLSS_INT32, false, 0, true, true, false, "qkv_stride_d3"},
                                                                                                 {12, MLSS_INT32, false, 0, true, true, false, "output_stride_d0"},
                                                                                                 {13, MLSS_INT32, false, 0, true, true, false, "output_stride_d1"},
                                                                                                 {14, MLSS_INT32, false, 0, true, true, false, "output_stride_d2"},
                                                                                                 {15, MLSS_INT32, false, 0, true, true, false, "output_stride_d3"}}};

} // namespace mlss::gqa::ck::wmma::fp16
