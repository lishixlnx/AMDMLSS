/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Integer arithmetic helper functions
    //=====================================================================================================================

    //---------------------------------------------------------------------------------------------------------------------
    // Divide value by divisor and round up to the nearest integer
    // Example: integer_divide_ceil(10, 3) = 4 (since 10/3 = 3.33... rounds up to 4)
    template <class T>
        requires std::is_integral_v<T>
    T integer_divide_ceil(const T& value, const std::type_identity_t<T>& divisor)
    {
        const T& div = divisor;

        if (div == 0)
        {
            return 0; // Avoid division by zero
        }

        if constexpr (std::is_unsigned_v<T>)
        {
            // For unsigned: (value + divisor - 1) / divisor
            return (value + div - 1) / div;
        }
        else
        {
            // For signed, handle negative values properly
            if (value >= 0)
            {
                return (value + div - 1) / div;
            }
            else
            {
                // Negative values: ceil(-10/3) = ceil(-3.33...) = -3
                return value / div;
            }
        }
    }

    //---------------------------------------------------------------------------------------------------------------------
    // Floor value to the nearest multiple of divisor (round down)
    // Example: integer_device_floor(10, 3) = 9 (largest multiple of 3 that's <= 10)
    template <class T>
        requires std::is_integral_v<T>
    T integer_divide_floor(const T& value, const std::type_identity_t<T>& divisor)
    {
        const T& div = divisor;

        if (div == 0)
        {
            return value; // No alignment possible
        }

        if constexpr (std::is_unsigned_v<T>)
        {
            // For unsigned: (value / divisor) * divisor
            return (value / div) * div;
        }
        else
        {
            // For signed, handle negative values properly
            if (value >= 0)
            {
                return (value / div) * div;
            }
            else
            {
                // For negative values: floor(-10/3)*3 = -4*3 = -12
                T quotient = value / div;
                T remainder = value % div;
                if (remainder != 0)
                {
                    quotient -= 1;
                }
                return quotient * div;
            }
        }
    }

    //---------------------------------------------------------------------------------------------------------------------
    // Round value to the nearest multiple of divisor
    // Example: integer_device_round(10, 3) = 9, integer_device_round(11, 3) = 12
    template <class T>
        requires std::is_integral_v<T>
    T integer_divide_round(const T& value, const std::type_identity_t<T>& divisor)
    {
        const T& div = divisor;

        if (div == 0)
        {
            return value; // No alignment possible
        }

        if constexpr (std::is_unsigned_v<T>)
        {
            // For unsigned: round to nearest multiple
            T remainder = value % div;
            T half = div / 2;

            if (remainder >= half + (div % 2)) // Handle odd divisors correctly
            {
                // Round up
                return value + (div - remainder);
            }
            else
            {
                // Round down
                return value - remainder;
            }
        }
        else
        {
            // For signed values
            if (value >= 0)
            {
                T remainder = value % div;
                T half = div / 2;

                if (remainder >= half + (div % 2))
                {
                    return value + (div - remainder);
                }
                else
                {
                    return value - remainder;
                }
            }
            else
            {
                // For negative values: round to nearest multiple
                T abs_value = -value;
                T remainder = abs_value % div;
                T half = div / 2;

                if (remainder >= half + (div % 2))
                {
                    return -(abs_value + (div - remainder));
                }
                else
                {
                    return -(abs_value - remainder);
                }
            }
        }
    }

} // namespace mlss
