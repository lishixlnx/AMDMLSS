/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Device and Architecture Utilities
    //=====================================================================================================================

    /// @brief Convert GPU codename to architecture flag
    GfxArchitectureFlags GpuCodenameToArchitectureFlag(GpuCodenameFlags codename);

    /// @brief Convert architecture flag to string
    std::string_view gfxArchitectureFlagsToString(GfxArchitectureFlags flag);

    /// @brief Convert GPU codename flag to string
    std::string_view gpuCodenameFlagsToString(GpuCodenameFlags flag);

    /// @brief Convert architecture string to flag
    std::expected<GfxArchitectureFlags, std::error_code> architechtureStringToFlag(std::string_view gfx);

    //=====================================================================================================================
    // Stream operators for flags
    //=====================================================================================================================

    template<class T, class Traits>
    std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GfxArchitectureFlags& flag);

    template<class T, class Traits>
    std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GpuCodenameFlags& flag);

} // namespace mlss

