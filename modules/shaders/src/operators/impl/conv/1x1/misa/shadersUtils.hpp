/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "core/core.hpp"
#include "shaders/shaders.hpp"
#include "../../utils.hpp"
#include "../../../opUtils.hpp"

namespace mlss::conv::one_by_one::misa
{

    mlss::op::utils::MetaCmdCaps isMisaShadersAvailable(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params);

    std::expected<Binaries, std::error_code> getMisaShadersBlob(const GfxIpTriple& gfxip, const mlss::conv::utils::GenericConvParams& params);

} // namespace mlss::conv::one_by_one::misa
