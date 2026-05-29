#include "api_backend.hpp"
#include "amdmlss/amdmlss_api.h"
#include "core/core.hpp"
#include "shaders/shaders.hpp"

#include <string_view>
#include <memory>
#include <format>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR  // windows.h defines ERROR=0 which clashes with mlss::VerboseLevel::ERROR

// SEH shim helpers — __try/__except cannot be used inside lambdas or functions
// with C++ object unwinding on MSVC/Clang, so each dangerous call site gets
// its own plain-C-style shim function that wraps exactly one MLSS C-API call.
// If an access violation or similar structured exception occurs inside MLSS,
// the shim returns MLSS_ERROR_FAILURE instead of crashing the host process.

static MLSSstatus seh_createContext(MLSScontext* ctx, const MLSSstring asic, const MLSSstring op) noexcept
{
    __try { return mlss::setLastError(mlss::createContext(*ctx, asic, op, nullptr)); }
    __except(EXCEPTION_EXECUTE_HANDLER) { return MLSS_ERROR_FAILURE; }
}

static MLSSstatus seh_getCaps(const MLSScontext ctx, MLSSstatus** ps, MLSSsize* ns) noexcept
{
    __try { return mlss::setLastError(mlss::getCaps(ctx, ps, ns)); }
    __except(EXCEPTION_EXECUTE_HANDLER) { return MLSS_ERROR_FAILURE; }
}

struct SetParamArgs { MLSScontext* ctx; const MLSSstring op; MLSSenum flag; const MLSSvoid* val; };
static MLSSstatus seh_setParamByEnum(SetParamArgs a) noexcept
{
    __try {
        if (!mlss::setParams(a.ctx, a.op, a.flag, a.val))
            return mlss::setLastError(MLSS_ERROR_PARAMETER_NOT_FOUND);
        return mlss::setLastError(MLSS_SUCCESS);
    }
    __except(EXCEPTION_EXECUTE_HANDLER) { return MLSS_ERROR_FAILURE; }
}

static MLSSstatus seh_getBinariesEx(const MLSScontext ctx, MLSSbinary** bins,
                                     MLSSsize* n, MLSSbinaryKind kind) noexcept
{
    __try { return mlss::setLastError(mlss::createBinariesEx(*bins, ctx, n, kind)); }
    __except(EXCEPTION_EXECUTE_HANDLER) { return MLSS_ERROR_FAILURE; }
}
#endif

#ifndef MLSS_VERSION_MAJOR
#define MLSS_VERSION_MAJOR 0
#endif

#ifndef MLSS_VERSION_MEDIAN
#define MLSS_VERSION_MEDIAN 0
#endif

#ifndef MLSS_VERSION_MINOR
#define MLSS_VERSION_MINOR 0
#endif

namespace
{
    const char* const ERROR_STRINGS[] =
        {
            "Success",
            "Failure",
            "Configuration not supported",
            "Graphix not supported",
            "Operator not supported",
            "Invalid parameter",
            "Parameter not found",
            "Operator not found",
            "Bad memory allocation",
            "Context update failed",
            "Context not created",
            "Unknown attribute",
            "Context already existing"};

    constexpr size_t ERROR_STRINGS_SIZE = sizeof(ERROR_STRINGS) / sizeof(ERROR_STRINGS[0]);
} // namespace

extern "C"
{
    // Error handling functions

    MLSSenum mlssPeakAtLastError()
    {
        return mlss::returnLastError();
    }

    MLSSenum mlssGetLastError()
    {
        return mlss::resetLastError();
    }

    MLSSstring mlssGetErrorString(const MLSSenum error)
    {
        size_t index = static_cast<size_t>(error);
        if (index < ERROR_STRINGS_SIZE)
        {
            return const_cast<MLSSstring>(ERROR_STRINGS[index]);
        }

        return const_cast<MLSSstring>("Unknown error");
    }

    // Version functions
    MLSSstatus mlssGetVersion(MLSSuint32* const major, MLSSuint32* const median, MLSSuint32* const minor, MLSSstringref flavor)
    {
        if (major != nullptr)
        {
            *major = MLSS_VERSION_MAJOR;
        }

        if (median != nullptr)
        {
            *median = MLSS_VERSION_MEDIAN;
        }

        if (minor != nullptr)
        {
            *minor = MLSS_VERSION_MINOR;
        }

        if (flavor != nullptr)
        {
            *flavor = const_cast<MLSSstring>(MLSS_FLAVOR);
        }

        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSstring mlssGetVersionAsString()
    {
        // Static buffer for thread safety concerns
        static thread_local char buffer[64];

        MLSSuint32 minor = 0, median = 0, major = 0;
        MLSSstring rev = nullptr;

        mlssGetVersion(&major, &median, &minor, &rev);

        std::string tmp = std::format("{}.{}.{}-{}", major, median, minor, rev ? rev : "unknown");

        // Copy to static buffer to ensure persistence using std::string::copy
        size_t copy_length = std::min(tmp.size(), static_cast<size_t>(63));
        tmp.copy(buffer, copy_length);
        buffer[copy_length] = '\0'; // Ensure null termination

        return buffer;
    }

    // Context management

    MLSSstatus mlssCreateContext(MLSScontext* context, const MLSSstring asic, const MLSSstring opName)
    {
        if (context == 0)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

#ifdef _WIN32
        return seh_createContext(context, asic, opName);
#else
        return mlss::setLastError(mlss::createContext(*context, asic, opName, nullptr));
#endif
    }

    MLSSstatus mlssCreateContextList(MLSScontext* context, const MLSSstring asic, const MLSSstring opName, ...)
    {
        if (context == 0)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        va_list args;
        va_start(args, opName);

        if (auto status = mlss::createContext(*context, asic, opName, &args); status != MLSS_SUCCESS)
        {
            return mlss::setLastError(status);
        }

        va_end(args);

        return mlss::setLastError(MLSS_SUCCESS);
    }

    // get caps
    MLSSstatus mlssGetCaps(const MLSScontext context, MLSSstatus** const pStatuses, MLSSsize* const nStatuses)
    {
#ifdef _WIN32
        return seh_getCaps(context, pStatuses, nStatuses);
#else
        return mlss::setLastError(mlss::getCaps(context, pStatuses, nStatuses));
#endif
    }

    // Binary management

    MLSSstatus mlssGetBinaries(const MLSScontext context, MLSSbinary** const binaries, MLSSsize* const numBinaries)
    {
        if (context == 0)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        return mlss::setLastError(mlss::createBinaries(*binaries, context, numBinaries));
    }

    MLSSstatus mlssGetBinariesEx(const MLSScontext context, MLSSbinary** const binaries,
                                  MLSSsize* const numBinaries, MLSSbinaryKind kind)
    {
        if (context == 0)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

#ifdef _WIN32
        return seh_getBinariesEx(context, binaries, numBinaries, kind);
#else
        return mlss::setLastError(mlss::createBinariesEx(*binaries, context, numBinaries, kind));
#endif
    }

    // Operator management

    MLSSvoid mlssEnumerateOperators(MLSSstringarray operators, MLSSint32* count)
    {
        (MLSSvoid) operators;
        (MLSSvoid) count;

        mlss::not_implemented_as_a_warning();
        mlss::setLastError(MLSS_WARNING_NOT_IMPLEMENTED);
    }

    // Parameter management for binary creation

    MLSSstatus mlssGetParameterByName(const MLSScontext context, const MLSSstring opName, const MLSSstring parameterName, MLSSvoid* value)
    {
        (MLSSvoid) context;
        (MLSSvoid) opName;
        (MLSSvoid) parameterName;
        (MLSSvoid) value;
        mlss::not_implemented_as_a_warning();
        return mlss::setLastError(MLSS_WARNING_NOT_IMPLEMENTED);
    }

    MLSSstatus mlssGetParameterByEnum(const MLSScontext context, const MLSSstring opName, const MLSSenum parameterFlag, MLSSvoid* value)
    {
        (MLSSvoid) context;
        (MLSSvoid) opName;
        (MLSSvoid) parameterFlag;
        (MLSSvoid) value;
        mlss::not_implemented_as_a_warning();
        return mlss::setLastError(MLSS_WARNING_NOT_IMPLEMENTED);
    }

    MLSSstatus mlssSetParameterByName(MLSScontext* const context, const MLSSstring opName, const MLSSstring parameterName, const MLSSvoid* const value)
    {
        if ((context == 0) || (opName == nullptr))
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        if (!mlss::setParams(context, opName, parameterName, value))
        {
            return mlss::setLastError(MLSS_ERROR_PARAMETER_NOT_FOUND);
        }

        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSstatus mlssSetParameterByEnum(MLSScontext* const context, const MLSSstring opName, const MLSSenum parameterFlag, const MLSSvoid* const value)
    {

        if ((context == 0) || (opName == nullptr))
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

#ifdef _WIN32
        return seh_setParamByEnum({context, opName, parameterFlag, value});
#else
        if (!mlss::setParams(context, opName, parameterFlag, value))
            return mlss::setLastError(MLSS_ERROR_PARAMETER_NOT_FOUND);
        return mlss::setLastError(MLSS_SUCCESS);
#endif
    }

    MLSSstatus mlssPrintParameters(const MLSScontext context, const MLSSstring opName)
    {
        return mlss::printParams(context, opName);
    }

    MLSSstatus mlssPrintBinaries(const MLSSbinary* const binaries, const MLSSsize n)
    {
        return mlss::printBinaries(binaries, n);
    }

    MLSSstatus mlssVectorRetrieveData(const MLSSvector vector,
                                      MLSSvoid** const data,
                                      MLSSsize* const n,
                                      MLSSenum* const type)
    {
        return mlss::retrieveVectorData(vector, data, n, type);
    }

    MLSSbool mlssVectorIsValid(const MLSSvector vector)
    {
        return mlss::isVectorValid(vector);
    }

    // Verbose mode management

    MLSSstatus mlssSetVerboseLevel(MLSSenum level)
    {
        // Convert MLSSenum to VerboseLevel
        mlss::VerboseLevel verboseLevel = mlss::VerboseLevel::NONE;

        switch (level)
        {
            case 0:
                verboseLevel = mlss::VerboseLevel::NONE;
                break;
            case 1:
                verboseLevel = mlss::VerboseLevel::ERROR;
                break;
            case 2:
                verboseLevel = mlss::VerboseLevel::WARNING;
                break;
            case 3:
                verboseLevel = mlss::VerboseLevel::INFO;
                break;
            case 4:
                verboseLevel = mlss::VerboseLevel::DEBUG;
                break;
            case 5:
                verboseLevel = mlss::VerboseLevel::TRACE;
                break;
            default:
                return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        mlss::VerboseManager::getInstance().setLevel(verboseLevel);
        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSenum mlssGetVerboseLevel()
    {
        mlss::VerboseLevel level = mlss::VerboseManager::getInstance().getLevel();
        return static_cast<MLSSenum>(level);
    }

    MLSSstatus mlssEnableVerboseMode()
    {
        mlss::VerboseManager::getInstance().setLevel(mlss::VerboseLevel::INFO);
        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSstatus mlssDisableVerboseMode()
    {
        mlss::VerboseManager::getInstance().setLevel(mlss::VerboseLevel::NONE);
        return mlss::setLastError(MLSS_SUCCESS);
    }

    // Device query functions

    MLSSstatus mlssGetDeviceFeatures(MLSSdevicefeatures** devices, MLSSsize* numDevices)
    {
        if (!devices || !numDevices)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        auto result = mlss::getDeviceFeatures();
        if (!result.has_value())
        {
            *devices = nullptr;
            *numDevices = 0;
            return mlss::setLastError(MLSS_ERROR_FAILURE);
        }

        const auto& deviceVector = result.value();
        *numDevices = deviceVector.size();

        if (*numDevices == 0)
        {
            *devices = nullptr;
            return mlss::setLastError(MLSS_SUCCESS);
        }

        // Create a collection to store device features with proper lifetime management
        struct DeviceFeatureCollection_t
        {
            std::vector<MLSSdevicefeatures> devices;
            std::vector<std::string> gfxStrings;
        };

        auto collection = std::make_unique<DeviceFeatureCollection_t>();
        collection->devices.reserve(deviceVector.size());
        collection->gfxStrings.reserve(deviceVector.size());

        // Copy device features
        for (const auto& device : deviceVector)
        {
            collection->devices.push_back(device);
            collection->gfxStrings.push_back(device.m_gfx);

            // Update the pointer to point to our stored string
            collection->devices.back().m_gfx = collection->gfxStrings.back().c_str();
        }

        // Store the collection in MemoryManager using Any
        mlss::Any collection_any = std::move(collection);
        MLSShandle handle = mlss::MemoryManager::addObject(std::move(collection_any));

        // Mark as initialized
        mlss::Any* stored_any = mlss::MemoryManager::template getPointer<mlss::Any>(handle);
        if (!stored_any)
        {
            *devices = nullptr;
            *numDevices = 0;
            return mlss::setLastError(MLSS_ERROR_BAD_MEMORY_ALLOCATION);
        }

        mlss::MemoryManager::markAsInitialized(&stored_any);

        // Get the stored collection to return pointer to its data
        if (mlss::anyIs<std::unique_ptr<DeviceFeatureCollection_t>>(*stored_any))
        {
            auto& stored_collection = mlss::anyCast<std::unique_ptr<DeviceFeatureCollection_t>&>(*stored_any);
            *devices = stored_collection->devices.data();
        }
        else
        {
            *devices = nullptr;
            *numDevices = 0;
            return mlss::setLastError(MLSS_ERROR_BAD_MEMORY_ALLOCATION);
        }

        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSstatus mlssGetOptimalDeviceFeatures(MLSSdevicefeatures* device)
    {
        if (!device)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        auto result = mlss::getOptimalDeviceFeatures();
        if (!result.has_value())
        {
            return mlss::setLastError(MLSS_ERROR_FAILURE);
        }

        // Create a structure to hold the device and its string
        struct OptimalDeviceHolder_t
        {
            MLSSdevicefeatures device;
            std::string gfxString;
        };

        auto holder = std::make_unique<OptimalDeviceHolder_t>();
        holder->device = result.value();
        holder->gfxString = result.value().m_gfx;
        holder->device.m_gfx = holder->gfxString.c_str();

        // Copy to output
        *device = holder->device;

        // Store the holder in MemoryManager to keep the string alive
        mlss::Any holder_any = std::move(holder);
        MLSShandle handle = mlss::MemoryManager::addObject(std::move(holder_any));

        // Mark as initialized
        mlss::Any* stored_any = mlss::MemoryManager::template getPointer<mlss::Any>(handle);
        if (stored_any)
        {
            mlss::MemoryManager::markAsInitialized(&stored_any);
        }

        return mlss::setLastError(MLSS_SUCCESS);
    }

    // Custom Type Registration Functions

    // Global registry for custom types
    static std::unordered_map<MLSSenum, MLSScustomtypeinfo> g_customTypeRegistry;
    static MLSSenum g_nextCustomTypeId = MLSS_CUSTOM_TYPE_START;
    static std::mutex g_customTypeRegistryMutex;

    // String storage for custom type names and operators (owns the memory)
    struct CustomTypeStringStorage
    {
        std::string typeName;
        std::vector<std::string> supportedOperators;
    };
    static std::unordered_map<MLSSenum, CustomTypeStringStorage> g_customTypeStrings;

    MLSSenum mlssRegisterCustomType(const MLSSstring typeName,
                                    MLSScustomtypeset setFunc,
                                    MLSScustomtypeget getFunc,
                                    ...)
    {
        if (!typeName || !setFunc || !getFunc)
        {
            mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
            return 0;
        }

        std::lock_guard<std::mutex> lock(g_customTypeRegistryMutex);

        // Check if type name already exists
        for (const auto& [id, info] : g_customTypeRegistry)
        {
            if (info.m_typeName && std::strcmp(info.m_typeName, typeName) == 0)
            {
                mlss::setLastError(MLSS_ERROR_ALREADY_EXISTING_CONTEXT);
                return 0;
            }
        }

        // Collect supported operators from variadic arguments
        std::vector<std::string> supportedOps;
        va_list args;
        va_start(args, getFunc);

        const char* op = va_arg(args, const char*);
        while (op && std::strcmp(op, MLSS_END_LIST) != 0)
        {
            supportedOps.push_back(op);
            op = va_arg(args, const char*);
        }
        va_end(args);

        // Create custom type info
        MLSScustomtypeinfo info;
        info.m_typeId = g_nextCustomTypeId++;
        info.m_setFunc = setFunc;
        info.m_getFunc = getFunc;
        info.m_printFunc = nullptr; // Optional, can be added later

        // Store strings in our storage (owns the memory)
        CustomTypeStringStorage& storage = g_customTypeStrings[info.m_typeId];
        storage.typeName = typeName;
        storage.supportedOperators = std::move(supportedOps);

        // Point the C struct to our owned strings
        info.m_typeName = storage.typeName.c_str();

        // Create pointer array for supported operators
        if (!storage.supportedOperators.empty())
        {
            info.m_supportedOperators = new char*[storage.supportedOperators.size() + 1];
            for (size_t i = 0; i < storage.supportedOperators.size(); ++i)
            {
                // Cast away const since the C API expects char** but we won't modify
                info.m_supportedOperators[i] = const_cast<char*>(storage.supportedOperators[i].c_str());
            }
            info.m_supportedOperators[storage.supportedOperators.size()] = nullptr; // NULL-terminate
        }
        else
        {
            info.m_supportedOperators = nullptr;
        }

        // Register the type
        g_customTypeRegistry[info.m_typeId] = info;

        mlss::setLastError(MLSS_SUCCESS);
        return info.m_typeId;
    }

    MLSSstatus mlssGetCustomTypeInfo(MLSSenum customTypeId, MLSScustomtypeinfo* info)
    {
        if (!info)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        std::lock_guard<std::mutex> lock(g_customTypeRegistryMutex);

        auto it = g_customTypeRegistry.find(customTypeId);
        if (it == g_customTypeRegistry.end())
        {
            return mlss::setLastError(MLSS_ERROR_PARAMETER_NOT_FOUND);
        }

        *info = it->second;
        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSstatus mlssUnregisterCustomType(MLSSenum customTypeId)
    {
        std::lock_guard<std::mutex> lock(g_customTypeRegistryMutex);

        auto it = g_customTypeRegistry.find(customTypeId);
        if (it == g_customTypeRegistry.end())
        {
            return mlss::setLastError(MLSS_ERROR_PARAMETER_NOT_FOUND);
        }

        // Clean up allocated memory
        MLSScustomtypeinfo& info = it->second;

        // Delete the pointer array (strings are owned by g_customTypeStrings)
        if (info.m_supportedOperators)
        {
            delete[] info.m_supportedOperators;
        }

        // Remove from registries
        g_customTypeStrings.erase(info.m_typeId);
        g_customTypeRegistry.erase(it);

        return mlss::setLastError(MLSS_SUCCESS);
    }

    MLSSbool mlssIsCustomType(MLSSenum typeId)
    {
        std::lock_guard<std::mutex> lock(g_customTypeRegistryMutex);
        return g_customTypeRegistry.find(typeId) != g_customTypeRegistry.end();
    }

    MLSSstatus mlssSetParameterByNameTyped(MLSScontext* const context,
                                           const MLSSstring opName,
                                           const MLSSstring parameterName,
                                           MLSSenum valueType,
                                           const MLSSvoid* const value)
    {
        if (!context || !opName || !parameterName || !value)
        {
            return mlss::setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        // Check if it's a custom type
        if (mlssIsCustomType(valueType))
        {
            std::lock_guard<std::mutex> lock(g_customTypeRegistryMutex);

            auto it = g_customTypeRegistry.find(valueType);
            if (it != g_customTypeRegistry.end() && it->second.m_setFunc)
            {
                // Use the custom type's set function
                return it->second.m_setFunc(context, opName, value);
            }
        }

        // Fall back to regular parameter setting for built-in types
        return mlssSetParameterByName(context, opName, parameterName, value);
    }

} // extern "C"
