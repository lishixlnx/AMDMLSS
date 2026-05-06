/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // ELF Binary Utilities
    //=====================================================================================================================

    /// @brief Extract kernel name from ELF binary
    /// @param arr ELF binary bytes
    /// @return Name of the kernel found in the ELF binary, or
    ///         MLSSErrorCode::InvalidElfBinary if the buffer is not a
    ///         valid AMDGPU ELF or does not contain exactly one kernel.
    [[nodiscard]] std::expected<std::string, std::error_code> getKernelName(
        const std::span<const std::uint8_t>& arr);

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::error_code> getNonRelocatable(const std::span<const std::uint8_t>& arr,
         const GfxIpTriple& gfxIpHighEnd, const GfxIpTriple& gfxIpTarget);

    /// @brief Extract the required workgroup size from ELF kernel metadata via amd_comgr.
    /// @param arr ELF binary bytes (relocatable or executable)
    /// @return MLSSdim3 with the workgroup dimensions, or an error code
    [[nodiscard]] std::expected<MLSSdim3, std::error_code> getWorkgroupSize(
        const std::span<const std::uint8_t>& arr);

} // namespace mlss
