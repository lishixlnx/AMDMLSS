/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#include "core/core.hpp"

namespace mlss
{
    bool calcHyperboloid(const HyperConsts& consts, const float& c, const float& k, const float& d)
    {
        const float deltaC = std::max(c - static_cast<float>(consts.cOffset), 0.0f);
        const float deltaK = std::max(k - static_cast<float>(consts.kOffset), 0.0f);
        const float deltaD = std::max(d - static_cast<float>(consts.xyOffset * consts.xyOffset), 0.0f);

        const float distCubit = std::pow(static_cast<float>(consts.dist), 3.0f);

        return (deltaC * deltaK * deltaD - distCubit) >= 0.0f;
    }

} // namespace mlss
