/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // ELF Binary Utilities
    //=====================================================================================================================

    /// @brief Extract kernel name from ELF binary
    /// @param arr ELF binary bytes
    /// @return Name of the kernel found in the ELF binary
    /// @throws std::runtime_error if no kernel or multiple kernels are found
    [[nodiscard]] std::string getKernelName(const std::span<const std::uint8_t>& arr);

    [[nodiscard]] std::expected<std::vector<std::uint8_t>, std::error_code> getNonRelocatable(const std::span<const std::uint8_t>& arr,
         const GfxIpTriple& gfxIpHighEnd, const GfxIpTriple& gfxIpTarget);

} // namespace mlss
