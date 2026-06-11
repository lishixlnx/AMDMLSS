/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "core/core.hpp"
// Binary shader payloads use mlss::StaticShaderType, ShaderDescriptor, and make_binary_blob (see shaders/shaders.hpp).
#include "shaders/shaders.hpp"

namespace mlss::mha::ck::wmma
{

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct);

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct);

} // namespace mlss::mha::ck::wmma
