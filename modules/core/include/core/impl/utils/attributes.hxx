/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Attribute Utilities
    //=====================================================================================================================

    /// @brief Get attribute name from enum value
    std::string getAttributeNameFromEnum(const std::uint32_t& flag);

    /// @brief Create attributes for a given operator
    void createAttributes(const std::string& opName, std::vector<Attribute>& attributes);

    /// @brief Create attribute with specific range and type
    template<class T>
    Attribute makeAttribute(const std::string& range, const std::uint32_t& attr_enum, const std::uint32_t& type_enum);

    /// @brief Create attribute with specific range and type (enum64 version)
    template<class T>
    Attribute makeAttribute(const std::string& range, const std::uint32_t& attr_enum, const enum64& type_enum);

    //=====================================================================================================================
    // Stream operator for Attribute
    //=====================================================================================================================

    template<class CharT, class Traits>
    std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const Attribute& obj);

} // namespace mlss

