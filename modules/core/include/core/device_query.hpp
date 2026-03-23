/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <vector>
#include <string>
#include <expected>
#include <system_error>
#include "amdmlss/amdmlss_api_cdefs.h"

namespace mlss
{
    /**
     * @brief Query all available GPU devices and return their features
     * 
     * This function enumerates all GPU devices available on the system and
     * returns a vector containing the features of each usable device.
     * 
     * @return std::expected<std::vector<MLSSdevice_features>, std::error_code> Vector of device features for each usable GPU device, or error code
     * @note Returns an error if HIP runtime is not available
     */
    std::expected<std::vector<MLSSdevicefeatures>, std::error_code> getDeviceFeatures();
    
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
    std::expected<MLSSdevicefeatures, std::error_code> getOptimalDeviceFeatures();
}
