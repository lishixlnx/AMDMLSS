/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Enum and Type Flag Utilities
    //=====================================================================================================================

    /// @brief Create an array enum from type and size
    [[nodiscard]] enum64 makeArrayEnum(const std::uint32_t& type, const size_t& size = 1);

    /// @brief Get the size from an enum64 type flag
    [[nodiscard]] size_t getArrayEnumSize(const enum64& type);

    /// @brief Get the type from an enum64 type flag
    [[nodiscard]] std::uint32_t getArrayEnumType(const enum64& type);

    /// @brief Get the element type from an enum64 type flag
    [[nodiscard]] std::uint32_t getArrayEnumElementType(const enum64& type);

    /// @brief Convert string to enum flag
    [[nodiscard]] enum64 getFlagFromString(const std::string& src);

    /// @brief Get type flag for a given C++ type
    template <class T>
    [[nodiscard]] enum64 getTypeFlag();

    //=====================================================================================================================
    [[nodiscard]] std::expected<OperatorFlag, std::error_code> getOperatorFlagsFromString(std::string_view src) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<GfxIpTriple, std::error_code> gpuCodenameToGfxIpTriple(GpuCodenameFlags codename) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<std::string_view, std::error_code> gfxIpTripleToString(GfxIpTriple gfxIp) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<std::string_view, std::error_code> gpuCodenameFlagsToString(GpuCodenameFlags flag) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<GfxIpTriple, std::error_code> architectureStringToGfxIpTriple(std::string_view gfx) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<GpuCodenameFlags, std::error_code> gpuCodenameStringToFlag(std::string_view codename) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<GfxIpTriple, std::error_code> elfMatchToGfxIpTriple(const std::uint8_t& elfMatch) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<std::uint8_t, std::error_code> gfxIpTripleToElfMatch(GfxIpTriple gfxIp) noexcept;

    //=====================================================================================================================
    [[nodiscard]] bool isShaderSupported(ShaderTypesFlags shaderType, GfxIpTriple gfxIp) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<GfxIpTriple, std::error_code> getHighEndGfxIpTriple(GfxIpTriple gfxIp) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<std::int32_t, std::error_code> getNumCu(GfxIpTriple gfxIp) noexcept;

    //=====================================================================================================================
    [[nodiscard]] std::expected<std::int32_t, std::error_code> getMALL(GfxIpTriple gfxIp) noexcept;

    //=====================================================================================================================
    // Stream operators for flags
    //=====================================================================================================================

    template <class T, class Traits>
    std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GfxIpTriple& gfxIp);

    template <class T, class Traits>
    std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& os, const GpuCodenameFlags& flag);

} // namespace mlss
