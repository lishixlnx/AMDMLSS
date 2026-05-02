/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "core/core.hpp"
#include "shaders/shaders.hpp"
#include "../../../utils.hpp"
#include "../../../../opUtils.hpp"

namespace mlss::conv::one_by_one::hip::wmma
{

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct);

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct);

} // namespace mlss::conv::one_by_one::hip::wmma
