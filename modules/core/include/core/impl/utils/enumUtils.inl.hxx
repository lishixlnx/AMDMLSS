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


        //=====================================================================================================================
        template<class T, class Traits>
        std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GfxArchitectureFlags& flag)
        {
            namespace fs = std::filesystem;
    
            auto result = gfxArchitectureFlagsToString(flag);
            if (!result.has_value())
            {
                os << "Unknown";
                return os;
            }
            fs::path pth = std::string(result.value());
    
            if constexpr (std::is_same_v<T, std::string::value_type>)
            {
                os << pth.string();
            }
            else if constexpr (std::is_same_v<T, std::wstring::value_type>)
            {
                os << pth.wstring();
            }
            else if constexpr (std::is_same_v<T, std::u8string::value_type>)
            {
                os << pth.u8string();
            }
            else if constexpr (std::is_same_v<T, std::u16string::value_type>)
            {
                os << pth.u16string();
            }
            else if constexpr (std::is_same_v<T, std::u32string::value_type>)
            {
                os << pth.u32string();
            }
    
            return os;
        }
    
        //=====================================================================================================================
        template<class T, class Traits>
        std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GpuCodenameFlags& flag)
        {
            namespace fs = std::filesystem;
            auto result = gpuCodenameFlagsToString(flag);
            if (!result.has_value())
            {
                os << "Unknown";
                return os;
            }
            fs::path pth = std::string(result.value());
    
            if constexpr (std::is_same_v<T, std::string::value_type>)
            {
                os << pth.string();
            }
            else if constexpr (std::is_same_v<T, std::wstring::value_type>)
            {
                os << pth.wstring();
            }
            else if constexpr (std::is_same_v<T, std::u8string::value_type>)
            {
                os << pth.u8string();
            }
            else if constexpr (std::is_same_v<T, std::u16string::value_type>)
            {
                os << pth.u16string();
            }
            else if constexpr (std::is_same_v<T, std::u32string::value_type>)
            {
                os << pth.u32string();
            }
            return os;
        }

} // namespace mlss

