/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "core/core.hpp"
#include <hip/hip_runtime.h>

namespace mlss
{
    namespace
    {
        // Custom error category for MLSS device query errors
        class DeviceQueryErrorCategory : public std::error_category
        {
        public:

            const char* name() const noexcept override
            {
                return "mlss_device_query";
            }

            std::string message(int ev) const override
            {
                switch (ev)
                {
                    case 1:
                        return "HIP runtime error";
                    case 2:
                        return "No devices found";
                    case 3:
                        return "No supported devices found";
                    default:
                        return "Unknown device query error";
                }
            }
        };

        const DeviceQueryErrorCategory& deviceQueryErrorCategory()
        {
            static DeviceQueryErrorCategory instance;
            return instance;
        }

        std::error_code makeDeviceQueryError(int code)
        {
            return std::error_code(code, deviceQueryErrorCategory());
        }

        // Helper function to convert architecture string to MLSS GFX string
        std::string archToMlssGfx(const char* gcnArchName)
        {
            if (!gcnArchName) return MLSS_GFXAUTOFIND;

            std::string archStr(gcnArchName);

            // Extract the numeric part from strings like "gfx1100" or "gfx90a"
            if (archStr.find("gfx") == 0 && archStr.length() > 3)
            {
                // Remove "gfx" prefix
                std::string numStr = archStr.substr(3);

                // Parse the numeric part
                char* endPtr = nullptr;
                int archNum = std::strtol(numStr.c_str(), &endPtr, 10);

                // Map GCN architecture numbers to MLSS GFX strings
                switch (archNum)
                {
                    case 1010:
                        return MLSS_GFX1010;
                    case 1011:
                        return MLSS_GFX1011;
                    case 1012:
                        return MLSS_GFX1012;
                    case 1030:
                        return MLSS_GFX1030;
                    case 1031:
                        return MLSS_GFX1031;
                    case 1032:
                        return MLSS_GFX1032;
                    case 1034:
                        return MLSS_GFX1034;
                    case 1100:
                        return MLSS_GFX1100;
                    case 1101:
                        return MLSS_GFX1101;
                    case 1102:
                        return MLSS_GFX1102;
                    case 1103:
                        return MLSS_GFX1103;
                    case 1150:
                        return MLSS_GFX1150;
                    case 1151:
                        return MLSS_GFX1151;
                    case 1152:
                        return MLSS_GFX1152;
                    case 1153:
                        return MLSS_GFX1153;
                    case 1154:
                        return MLSS_GFX1154;
                    case 1170:
                        return MLSS_GFX1170;
                    case 1171:
                        return MLSS_GFX1171;
                    case 1200:
                        return MLSS_GFX1200;
                    case 1201:
                        return MLSS_GFX1201;
                    default:
                        // Return AUTOFIND for unsupported architectures
                        return MLSS_GFXAUTOFIND;
                }
            }

            return MLSS_GFXAUTOFIND;
        }
    } // namespace

    std::expected<std::vector<MLSSdevicefeatures>, std::error_code> getDeviceFeatures()
    {
        std::vector<MLSSdevicefeatures> devices;

        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);

        if (err != hipSuccess)
        {
            return std::unexpected(makeDeviceQueryError(1));
        }

        if (deviceCount == 0)
        {
            return devices; // Return empty vector
        }

        for (int i = 0; i < deviceCount; ++i)
        {
            hipDeviceProp_t props;
            err = hipGetDeviceProperties(&props, i);

            if (err != hipSuccess)
            {
                continue;
            }

            // Convert GCN architecture to MLSS GFX string
            std::string gfxString = archToMlssGfx(props.gcnArchName);

            // Skip unsupported architectures
            if (gfxString == MLSS_GFXAUTOFIND)
            {
                continue;
            }

            MLSSdevicefeatures features;
            features.m_numCUs = props.multiProcessorCount;
            features.m_sizeShaderMem = static_cast<std::uint32_t>(props.sharedMemPerBlock);

            // Use memory manager to store the GFX string
            Any gfxAny(gfxString);
            auto handle = MemoryManager::addObject(std::move(gfxAny));

            // Get the string from the Any object
            Any* anyPtr = MemoryManager::getPointer<Any>(handle);
            const std::string* storedString = anyCast<std::string>(anyPtr);
            features.m_gfx = storedString->c_str();

            // Check if this is a dedicated GPU (not integrated)
            features.m_isDedicated = (props.integrated == 0);
            features.m_gpuIdx = i;

            devices.push_back(features);
        }

        if (devices.empty())
        {
            return std::unexpected(makeDeviceQueryError(3));
        }

        return devices;
    }

    std::expected<MLSSdevicefeatures, std::error_code> getOptimalDeviceFeatures()
    {
        // First get all available devices
        auto devicesResult = getDeviceFeatures();
        if (!devicesResult.has_value())
        {
            return std::unexpected(devicesResult.error());
        }

        auto devices = devicesResult.value();
        if (devices.empty())
        {
            return std::unexpected(makeDeviceQueryError(2));
        }

        // Helper lambda to extract GFX level from string (e.g., "MLSS_GFX1100" -> 1100)
        auto getGfxLevel = [](const char* gfxStr) -> int
        {
            if (!gfxStr || std::strlen(gfxStr) < 8) return 0;

            // Skip "MLSS_GFX" prefix and convert the number
            const char* numStr = gfxStr + 8;
            char* endPtr = nullptr;
            int level = std::strtol(numStr, &endPtr, 10);

            // If conversion failed or not all characters were consumed, return 0
            if (endPtr == numStr || *endPtr != '\0') return 0;

            return level;
        };

        // Sort devices by:
        // 1. Dedicated GPUs first (m_isDedicated = true)
        // 2. Higher GFX architecture level
        std::sort(devices.begin(), devices.end(), [&getGfxLevel](const MLSSdevicefeatures& a, const MLSSdevicefeatures& b)
                  {
                // First priority: dedicated GPUs
                if (a.m_isDedicated != b.m_isDedicated)
                {
                    return a.m_isDedicated > b.m_isDedicated;
                }
                
                // Second priority: higher GFX level
                int levelA = getGfxLevel(a.m_gfx);
                int levelB = getGfxLevel(b.m_gfx);
                return levelA > levelB; });

        // Return the best device (first after sorting)
        const auto& optimal = devices[0];

        return optimal;
    }

    //=====================================================================================================================
    // GFX Architecture detection functions
    //=====================================================================================================================

    //=====================================================================================================================
    bool isGfx110x(const GfxIpTriple& gfx)
    {
        return (gfx == IP_GFX1100) || (gfx == IP_GFX1101) ||
               (gfx == IP_GFX1102) || (gfx == IP_GFX1103);
    }

    //=====================================================================================================================
    bool isGfx115x(const GfxIpTriple& gfx)
    {
        return (gfx == IP_GFX1150) || (gfx == IP_GFX1151) ||
               (gfx == IP_GFX1152) || (gfx == IP_GFX1153) ||
               (gfx == IP_GFX1154);
    }

    //=====================================================================================================================
    bool isGfx117x(const GfxIpTriple& gfx)
    {
        return (gfx == IP_GFX1170) || (gfx == IP_GFX1171);
    }

    //=====================================================================================================================
    bool isGfx120x(const GfxIpTriple& gfx)
    {
        return (gfx == IP_GFX1200) || (gfx == IP_GFX1201);
    }

    //=====================================================================================================================
    bool isGfx10(const GfxIpTriple& gfx)
    {
        return (gfx == IP_GFX1000) || (gfx == IP_GFX1010) ||
               (gfx == IP_GFX1011) || (gfx == IP_GFX1012) ||
               (gfx == IP_GFX1013) || (gfx == IP_GFX1020) ||
               (gfx == IP_GFX1030) || (gfx == IP_GFX1031) ||
               (gfx == IP_GFX1032) || (gfx == IP_GFX1033) ||
               (gfx == IP_GFX1034) || (gfx == IP_GFX1035) ||
               (gfx == IP_GFX1036) || (gfx == IP_GFX1050);
    }

    //=====================================================================================================================
    bool isGfx11(const GfxIpTriple& gfx)
    {
        return isGfx110x(gfx) || isGfx115x(gfx) || isGfx117x(gfx);
    }

    //=====================================================================================================================
    bool isGfx12(const GfxIpTriple& gfx)
    {
        return isGfx120x(gfx);
    }

    //=====================================================================================================================
    bool isGfx13(const GfxIpTriple& gfx)
    {
        // GFX13 not yet supported
        return false;
    }

    //=====================================================================================================================
    bool isGfx11Plus(const GfxIpTriple& gfx)
    {
        return isGfx11(gfx) || isGfx12Plus(gfx);
    }

    //=====================================================================================================================
    bool isGfx12Plus(const GfxIpTriple& gfx)
    {
        return isGfx12(gfx) || isGfx13(gfx);
    }

    bool isGfx12Max(const GfxIpTriple& gfx)
    {
        return isGfx10(gfx) || isGfx11(gfx) || isGfx12(gfx);
    }

    bool areGfxIpsCompatible(const GfxIpTriple& gfxIpHighEnd, const GfxIpTriple& gfxIpTarget)
    {    
        return (gfxIpHighEnd.major == gfxIpTarget.major) &&
               (gfxIpHighEnd.minor == gfxIpTarget.minor);
    }

    GfxIpTriple resolveElfPatchTarget(const GfxIpTriple& contextGfxip)
    {
        if (!isGfx110x(contextGfxip))
        {
            return contextGfxip;
        }

        const auto optimalDevice = getOptimalDeviceFeatures();
        if (!optimalDevice.has_value())
        {
            return contextGfxip;
        }

        const auto runtimeGfxip = architectureStringToGfxIpTriple(optimalDevice->m_gfx);
        if (!runtimeGfxip.has_value())
        {
            return contextGfxip;
        }

        if (isGfx115x(*runtimeGfxip))
        {
            return *runtimeGfxip;
        }

        return contextGfxip;
    }
} // namespace mlss
