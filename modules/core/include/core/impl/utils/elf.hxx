/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // ELF Binary Utilities
    //=====================================================================================================================

    /// @brief Extract kernel name from ELF binary
    /// @param ptr Pointer to ELF binary data
    /// @param size Size of the binary data
    /// @return Name of the kernel found in the ELF binary
    /// @throws std::runtime_error if no kernel or multiple kernels are found
    [[nodiscard]] std::string getKernelName(const std::byte* const ptr, const std::size_t size);

    /// @brief Extract kernel name from ELF binary (span overload)
    inline std::string getKernelName(const std::span<const std::byte>& arr);

} // namespace mlss
