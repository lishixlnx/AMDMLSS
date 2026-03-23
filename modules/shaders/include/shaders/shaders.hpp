/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <array>
#include <tuple>
#include <vector>
#include <memory>
#include <expected>
#include <type_traits>
#include <string>

#include "core/core.hpp"

#define MLSS_ASSERT_ALWAYS() assert(false);
#define MLSS_ASSERT(cdt) assert(cdt);

#define MLSS_MAKE_BLOB(var) mlss::shaders::makeBinaryBlob(var, #var)

namespace mlss
{

namespace shaders
{

//=====================================================================================================================
template<typename T, std::size_t size>
requires std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint8_t>
std::unique_ptr<Binaries::Blob> makeBinaryBlob(const T(&tab)[size], const std::string& name)
{
    using Blob = Binaries::Blob;

    if constexpr (std::is_same_v<T, std::uint8_t>)
    {
        return std::make_unique<Blob>(Blob{ tab, size, MLSS_BINARY_TYPE_ELF, 0, name });
    }
    else
    {
        return std::make_unique<Blob>(Blob{ tab, size, MLSS_BINARY_TYPE_IL, 0, name });
    }
}

//=====================================================================================================================
template<typename T, std::size_t size>
requires std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint8_t>
std::unique_ptr<Binaries::Blob> makeBinaryBlob(const std::array<T, size>& tab, const std::string& name)
{
    using Blob = Binaries::Blob;

    if constexpr (std::is_same_v<T, std::uint8_t>)
    {
        return std::make_unique<Blob>(Blob{ tab.data(), size, MLSS_BINARY_TYPE_ELF, 0, name });
    }
    else
    {
        return std::make_unique<Blob>(Blob{ tab.data(), size, MLSS_BINARY_TYPE_IL, 0, name });
    }
}

//=====================================================================================================================
std::uint32_t convertShaderErrorToEnum(std::error_code error);


} // shaders




namespace math
{
//=====================================================================================================================

    template<class D, class S>
    constexpr D reinterpret_as(S v)
    {
        return reinterpret_cast<D&>(v);
    }

} // math

} // mlss

