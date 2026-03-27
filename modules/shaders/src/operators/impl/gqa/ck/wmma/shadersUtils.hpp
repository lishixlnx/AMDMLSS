/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "core/core.hpp"
#include "shaders/shaders.hpp"

namespace mlss::shaders::gqa::ck::wmma
{

    bool isWmmaShadersAvailable(
        GfxArchitectureFlags gfxArch,
        const std::uint32_t& sizeHeads,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& dataType);

    std::expected<Binaries, std::error_code> getWmmaShadersBlob(
        GfxArchitectureFlags gfxArch,
        const std::uint32_t& batchSize,
        const std::uint32_t& qHeadCount,
        const std::uint32_t& kvHeadCount,
        const std::uint32_t& headDim,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        bool useStrides,
        const std::uint32_t& dataType);

} // namespace mlss::shaders::gqa::ck::wmma
