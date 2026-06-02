/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "core/core.hpp"
#include "shaders/shaders.hpp"
#include "shaders/src/operators/impl/opUtils.hpp"

namespace mlss::gemm::mxn::hip
{

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& ip,
                                                    const std::vector<Attribute>& attr,
                                                    const void* cstmStruct);

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip,
                                                            const std::vector<Attribute>& attr,
                                                            const void* cstmStruct);

} // namespace mlss::gemm::mxn::hip
