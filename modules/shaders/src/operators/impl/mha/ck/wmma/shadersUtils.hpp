#pragma once
#include "core/core.hpp"
// Binary shader payloads use mlss::shaders::StaticShaderType, ShaderDescriptor, and make_binary_blob (see shaders/shaders.hpp).
#include "shaders/shaders.hpp"

namespace mlss::shaders::mha::ck::wmma
{

    bool isWmmaShadersAvailable(
        GfxArchitectureFlags gfxArch,
        const std::uint32_t& sizeHeads,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        const std::uint32_t& dataType);

    std::expected<Binaries, std::error_code> getWmmaShadersBlob(
        GfxArchitectureFlags gfxArch,
        const std::uint32_t& batchSize,
        const std::uint32_t& headCount,
        const std::uint32_t& headDim,
        const std::uint32_t& kvSequenceLength,
        const std::uint32_t& qSequenceLength,
        const std::uint32_t& packing,
        const std::uint32_t& dataType);

} // namespace mlss::shaders::mha::ck::wmma
