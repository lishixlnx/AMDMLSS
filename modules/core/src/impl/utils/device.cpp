/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
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
                    case 1: return "HIP runtime error";
                    case 2: return "No devices found";
                    case 3: return "No supported devices found";
                    default: return "Unknown device query error";
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
                    case 1010: return MLSS_GFX1010;
                    case 1011: return MLSS_GFX1011;
                    case 1012: return MLSS_GFX1012;
                    case 1030: return MLSS_GFX1030;
                    case 1031: return MLSS_GFX1031;
                    case 1032: return MLSS_GFX1032;
                    case 1034: return MLSS_GFX1034;
                    case 1100: return MLSS_GFX1100;
                    case 1101: return MLSS_GFX1101;
                    case 1102: return MLSS_GFX1102;
                    case 1103: return MLSS_GFX1103;
                    case 1150: return MLSS_GFX1150;
                    case 1151: return MLSS_GFX1151;
                    case 1152: return MLSS_GFX1152;
                    case 1153: return MLSS_GFX1153;
                    case 1154: return MLSS_GFX1154;
                    case 1170: return MLSS_GFX1170;
                    case 1171: return MLSS_GFX1171;
                    case 1200: return MLSS_GFX1200;
                    case 1201: return MLSS_GFX1201;
                    default:
                        // Return AUTOFIND for unsupported architectures
                        return MLSS_GFXAUTOFIND;
                }
            }
            
            return MLSS_GFXAUTOFIND;
        }
    }

    std::expected<std::vector<MLSSdevicefeatures>, std::error_code> getDeviceFeatures()
    {
        std::vector<MLSSdevicefeatures> devices;
        
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        
        if (err != hipSuccess)
        {
            error_log << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
            return std::unexpected(makeDeviceQueryError(1));
        }
        
        if (deviceCount == 0)
        {
            warning_log << "No devices found" << std::endl;
            return devices; // Return empty vector
        }
        
        info_log << "Found " << deviceCount << " device(s)" << std::endl;
        
        for (int i = 0; i < deviceCount; ++i)
        {
            hipDeviceProp_t props;
            err = hipGetDeviceProperties(&props, i);
            
            if (err != hipSuccess)
            {
                error_log << "Failed to get properties for device " << i << ": " << hipGetErrorString(err) << std::endl;
                continue;
            }
            
            // Convert GCN architecture to MLSS GFX string
            std::string gfxString = archToMlssGfx(props.gcnArchName);
            
            // Skip unsupported architectures
            if (gfxString == MLSS_GFXAUTOFIND)
            {
                info_log << "Skipping device " << i << " with unsupported architecture: " << props.gcnArchName << std::endl;
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
            
            debug_log << "Device " << i << ": " << props.name
                << ", CUs: " << features.m_numCUs
                << ", Shared Mem: " << features.m_sizeShaderMem
                << ", GFX: " << gfxString
                << ", Dedicated: " << (features.m_isDedicated ? "Yes" : "No") << std::endl;
            
            devices.push_back(features);
        }
        
        if (devices.empty())
        {
            warning_log << "No supported HIP devices found" << std::endl;
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
            error_log << "No devices available to select optimal device" << std::endl;
            return std::unexpected(makeDeviceQueryError(2));
        }
        
        // Helper lambda to extract GFX level from string (e.g., "MLSS_GFX1100" -> 1100)
        auto getGfxLevel = [](const char* gfxStr) -> int {
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
        std::sort(devices.begin(), devices.end(), 
            [&getGfxLevel](const MLSSdevicefeatures& a, const MLSSdevicefeatures& b) {
                // First priority: dedicated GPUs
                if (a.m_isDedicated != b.m_isDedicated)
                {
                    return a.m_isDedicated > b.m_isDedicated;
                }
                
                // Second priority: higher GFX level
                int levelA = getGfxLevel(a.m_gfx);
                int levelB = getGfxLevel(b.m_gfx);
                return levelA > levelB;
            });
        
        // Return the best device (first after sorting)
        const auto& optimal = devices[0];
        
        info_log << "Selected optimal device: GPU " << optimal.m_gpuIdx
            << ", GFX: " << optimal.m_gfx
            << ", Dedicated: " << (optimal.m_isDedicated ? "Yes" : "No")
            << ", CUs: " << optimal.m_numCUs << std::endl;
        
        return optimal;
    }

    //=====================================================================================================================
    // GFX Architecture detection functions
    //=====================================================================================================================


    //=====================================================================================================================
    bool isGfx110x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1100) || (gfx == GfxArchitectureFlags::Gfx1101) ||
            (gfx == GfxArchitectureFlags::Gfx1102) || (gfx == GfxArchitectureFlags::Gfx1103);
    }

    //=====================================================================================================================
    bool isGfx115x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1150) || (gfx == GfxArchitectureFlags::Gfx1151) ||
            (gfx == GfxArchitectureFlags::Gfx1152) || (gfx == GfxArchitectureFlags::Gfx1153) ||
            (gfx == GfxArchitectureFlags::Gfx1154);
    }

    //=====================================================================================================================
    bool isGfx117x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1170) || (gfx == GfxArchitectureFlags::Gfx1171);
    }

    //=====================================================================================================================
    bool isGfx120x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1200) || (gfx == GfxArchitectureFlags::Gfx1201);
    }

    //=====================================================================================================================
    bool isGfx10(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1000) || (gfx == GfxArchitectureFlags::Gfx1010) ||
               (gfx == GfxArchitectureFlags::Gfx1011) || (gfx == GfxArchitectureFlags::Gfx1012) ||
               (gfx == GfxArchitectureFlags::Gfx1013) || (gfx == GfxArchitectureFlags::Gfx1020) ||
               (gfx == GfxArchitectureFlags::Gfx1030) || (gfx == GfxArchitectureFlags::Gfx1031) ||
               (gfx == GfxArchitectureFlags::Gfx1032) || (gfx == GfxArchitectureFlags::Gfx1033) ||
               (gfx == GfxArchitectureFlags::Gfx1034) || (gfx == GfxArchitectureFlags::Gfx1035) ||
               (gfx == GfxArchitectureFlags::Gfx1036) || (gfx == GfxArchitectureFlags::Gfx1050);
    }

    //=====================================================================================================================
    bool isGfx11(const GfxArchitectureFlags& gfx)
    {
        return isGfx110x(gfx) || isGfx115x(gfx) || isGfx117x(gfx);
    }

    //=====================================================================================================================
    bool isGfx12(const GfxArchitectureFlags& gfx)
    {
        return isGfx120x(gfx);
    }

    //=====================================================================================================================
    bool isGfx13(const GfxArchitectureFlags& gfx)
    {
        // GFX13 not yet supported
        return false;
    }

    //=====================================================================================================================
    bool isGfx11Plus(const GfxArchitectureFlags& gfx)
    {
        return isGfx11(gfx) || isGfx12Plus(gfx);
    }

    //=====================================================================================================================
    bool isGfx12Plus(const GfxArchitectureFlags& gfx)
    {
        return isGfx12(gfx) || isGfx13(gfx);
    }

    bool isGfx12Max(const GfxArchitectureFlags& gfx)
    {
        return isGfx10(gfx) || isGfx11(gfx) || isGfx12(gfx);
    }
}
