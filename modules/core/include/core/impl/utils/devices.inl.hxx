/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    template<class T, class Traits>
    std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GfxArchitectureFlags& flag)
    {
        namespace fs = std::filesystem;

        fs::path pth = gfxArchitectureFlagsToString(flag);

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
        fs::path pth = gpuCodenameFlagsToString(flag);

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

