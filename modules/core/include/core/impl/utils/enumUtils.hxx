/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Enum and Type Flag Utilities
    //=====================================================================================================================

    /// @brief Create an array enum from type and size
    enum64 makeArrayEnum(const std::uint32_t& type, const size_t& size = 1);

    /// @brief Get the size from an enum64 type flag
    size_t getArrayEnumSize(const enum64& type);

    /// @brief Get the type from an enum64 type flag
    std::uint32_t getArrayEnumType(const enum64& type);

    /// @brief Get the element type from an enum64 type flag
    std::uint32_t getArrayEnumElementType(const enum64& type);

    /// @brief Convert string to enum flag
    enum64 getFlagFromString(const std::string& src);

    /// @brief Get type flag for a given C++ type
    template<class T>
    enum64 getTypeFlag();

} // namespace mlss

