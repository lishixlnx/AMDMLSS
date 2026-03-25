/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    inline std::string getKernelName(const std::span<const std::byte>& arr)
    {
        return getKernelName(arr.data(), arr.size());
    }
} // namespace mlss

