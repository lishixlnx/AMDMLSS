/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // Device query functions
    //=====================================================================================================================

    /**
     * @brief Query all available GPU devices and return their features
     *
     * This function enumerates all GPU devices available on the system and
     * returns a vector containing the features of each usable device.
     *
     * @return std::expected<std::vector<MLSSdevice_features>, std::error_code> Vector of device features for each usable GPU device, or error code
     * @note Returns an error if HIP runtime is not available
     */
    [[nodiscard]] std::expected<std::vector<MLSSdevicefeatures>, std::error_code> getDeviceFeatures();

    /**
     * @brief Get the optimal GPU device based on GFX level and dedicated status
     *
     * This function queries all available devices, sorts them by:
     * 1. Dedicated GPUs are preferred over integrated ones
     * 2. Higher GFX architecture levels are preferred
     *
     * @return std::expected<MLSSdevice_features, std::error_code> The optimal device features, or error code
     * @note Returns an error if no suitable devices are found
     */
    [[nodiscard]] std::expected<MLSSdevicefeatures, std::error_code> getOptimalDeviceFeatures();

    //=====================================================================================================================
    // GFX Architecture detection functions
    //=====================================================================================================================

    /// @brief Check if the GFX architecture is in the GFX110x family (1100, 1101, 1102, 1103)
    [[nodiscard]] bool isGfx110x(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is in the GFX115x family (1150, 1151, 1152, 1153, 1154)
    [[nodiscard]] bool isGfx115x(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is in the GFX117x family (1170, 1171)
    [[nodiscard]] bool isGfx117x(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is in the GFX120x family (1200, 1201)
    [[nodiscard]] bool isGfx120x(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is in the GFX10x family
    [[nodiscard]] bool isGfx10(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is GFX11 (110x, 115x, or 117x)
    [[nodiscard]] bool isGfx11(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is GFX12 (120x)
    [[nodiscard]] bool isGfx12(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is GFX13 (not yet supported)
    [[nodiscard]] bool isGfx13(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is GFX11 or later
    [[nodiscard]] bool isGfx11Plus(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is GFX12 or later
    [[nodiscard]] bool isGfx12Plus(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is max GFX12
    [[nodiscard]] bool isGfx12Max(const GfxIpTriple& gfx);

    /// @brief Check if the GFX architecture is compatible with the target GFX architecture
    [[nodiscard]] bool areGfxIpsCompatible(const GfxIpTriple& gfxIpHighEnd, const GfxIpTriple& gfxIpTarget);

} // namespace mlss
