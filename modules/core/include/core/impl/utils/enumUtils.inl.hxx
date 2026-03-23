/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    template<class T>
    enum64 getTypeFlag()
    {
        using value_type = T;

        if constexpr (std::is_same_v<T, bool>)
        {
            return makeArrayEnum(MLSS_BOOL);
        }
        else if constexpr (std::is_same_v<T, std::int8_t>)
        {
            return makeArrayEnum(MLSS_INT8);
        }
        else if constexpr (std::is_same_v<T, std::uint8_t>)
        {
            return makeArrayEnum(MLSS_UINT8);
        }
        else if constexpr (std::is_same_v<T, std::int16_t>)
        {
            return makeArrayEnum(MLSS_INT16);
        }
        else if constexpr (std::is_same_v<T, std::uint16_t>)
        {
            return makeArrayEnum(MLSS_UINT16);
        }
        else if constexpr (std::is_same_v<T, std::int32_t>)
        {
            return makeArrayEnum(MLSS_INT32);
        }
        else if constexpr (std::is_same_v<T, std::uint32_t>)
        {

            return makeArrayEnum(MLSS_UINT32);
        }
        else if constexpr (std::is_same_v<T, std::int64_t>)
        {
            return makeArrayEnum(MLSS_INT64);
        }
        else if constexpr (std::is_same_v<T, std::uint64_t>)
        {
            return makeArrayEnum(MLSS_UINT64);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return makeArrayEnum(MLSS_FLOAT32);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return makeArrayEnum(MLSS_FLOAT64);
        }
        else if constexpr (std::is_same_v<T, char>)
        {
            return makeArrayEnum(MLSS_INT8);
        }
        else if constexpr (std::is_array_v<T>)
        {
            return makeArrayEnum(getTypeFlag<std::ranges::range_value_t<T>>().high(), std::extent_v<T>);
        }
        else if constexpr (std::ranges::range<T>)
        {
            return makeArrayEnum(getTypeFlag<std::ranges::range_value_t<T>>().high(), std::tuple_size_v<T>);
        }
        else if constexpr (std::is_same_v<T, MLSSarg>)
        {
            return makeArrayEnum(MLSS_ARG);
        }
        else if constexpr (std::is_same_v<T, MLSSdim3>)
        {
            return makeArrayEnum(MLSS_DIM3);
        }
        else if constexpr (std::is_same_v<T, Context>)
        {
            return makeArrayEnum(MLSS_CONTEXT);
        }
        else if constexpr (std::is_same_v<T, Binaries>)
        {
            return makeArrayEnum(MLSS_BINARY);
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            return makeArrayEnum(MLSS_STRING);
        }
        else
        {

            return makeArrayEnum(MLSS_UNKNOWN_TYPE);
        }
    }

} // namespace mlss

