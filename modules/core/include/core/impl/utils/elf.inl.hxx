/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    template<std::size_t size>
    constexpr std::string getKernelName(const std::array<std::uint8_t, size>& arr)
    {
        return getKernelName(arr.data(), arr.size());
    }

} // namespace mlss

