/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Compile-time Integer Math Utilities
    //=====================================================================================================================

    /// @brief Divide value by divisor and round up to the nearest integer
    /// @example integer_divide_ceil(10, 3) = 4 (since 10/3 = 3.33... rounds up to 4)
    template <class T>
        requires std::is_integral_v<T>
    T integer_divide_ceil(const T& value, const std::type_identity_t<T>& divisor);

    /// @brief Floor value to the nearest multiple of divisor (round down)
    /// @example integer_divide_floor(10, 3) = 9 (largest multiple of 3 that's <= 10)
    template <class T>
        requires std::is_integral_v<T>
    T integer_divide_floor(const T& value, const std::type_identity_t<T>& divisor);

    /// @brief Round value to the nearest multiple of divisor
    /// @example integer_divide_round(10, 3) = 9, integer_divide_round(11, 3) = 12
    template <class T>
        requires std::is_integral_v<T>
    T integer_divide_round(const T& value, const std::type_identity_t<T>& divisor);

} // namespace mlss
