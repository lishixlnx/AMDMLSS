/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include <stdarg.h>

#include "amdmlss/amdmlss_export.h"
#include "amdmlss/amdmlss_api_cdefs.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // Error handling functions

    /**
     * @brief Peek at the last error without clearing it
     *
     * This function returns the last error that occurred in the AMDMLSS library
     * without clearing the error state. This allows checking for errors without
     * affecting subsequent error handling.
     *
     * @return MLSSenum The error code of the last error that occurred
     * @note This function does not clear the error state
     * @see mlssGetLastError() for retrieving and clearing the error
     */
    MLSS_API MLSSenum mlssPeakAtLastError();

    /**
     * @brief Get and clear the last error
     *
     * This function returns the last error that occurred in the AMDMLSS library
     * and clears the error state. After calling this function, the error state
     * will be reset to MLSS_STATUS_SUCCESS.
     *
     * @return MLSSenum The error code of the last error that occurred
     * @note This function clears the error state after retrieval
     * @see mlssPeakAtLastError() for checking errors without clearing
     */
    MLSS_API MLSSenum mlssGetLastError();

    /**
     * @brief Get a human-readable string description of an error code
     *
     * Converts an AMDMLSS error code into a descriptive string that can be
     * used for error reporting and debugging purposes.
     *
     * @param error The error code to convert to a string
     * @return MLSSstring A string describing the error, or "Unknown error" if the code is invalid
     * @note The returned string is statically allocated and should not be freed
     */
    MLSS_API MLSSstring mlssGetErrorString(const MLSSenum error);

    // Version functions

    /**
     * @brief Get the version information of the AMDMLSS library
     *
     * Retrieves the major, median (minor), and patch version numbers of the
     * AMDMLSS library, along with an optional flavor string (e.g., "release", "debug").
     *
     * @param[out] major Pointer to store the major version number (can be NULL)
     * @param[out] median Pointer to store the median (minor) version number (can be NULL)
     * @param[out] minor Pointer to store the minor (patch) version number (can be NULL)
     * @param[out] flavor Reference to store the version flavor string (can be NULL)
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note Any of the output parameters can be NULL if that information is not needed
     */
    MLSS_API MLSSstatus mlssGetVersion(MLSSuint32* const major, MLSSuint32* const median, MLSSuint32* const minor, MLSSstringref flavor);

    /**
     * @brief Get the version information as a formatted string
     *
     * Returns a human-readable string containing the complete version information
     * of the AMDMLSS library in the format "major.median.minor-flavor".
     *
     * @return MLSSstring A string containing the version information
     * @note The returned string is statically allocated and should not be freed
     */
    MLSS_API MLSSstring mlssGetVersionAsString();

    // Context management

    /**
     * @brief Create a new AMDMLSS context for a specific operation
     *
     * Creates a context that encapsulates the state for executing a specific
     * operation on a given ASIC (GPU architecture). The context must be used
     * for all subsequent operations related to this specific task.
     *
     * @param[out] context Pointer to store the created context handle
     * @param asic String identifying the target ASIC/GPU architecture (e.g., "gfx1100", "gfx1201")
     * @param opName String identifying the operation type (e.g., "gemm", "conv", "mha")
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The created context must be properly managed and used for subsequent operations
     * @warning The context should be properly cleaned up when no longer needed
     */
    MLSS_API MLSSstatus mlssCreateContext(MLSScontext* context, const MLSSstring asic, const MLSSstring opName);

    /**
     * @brief Create a new AMDMLSS context for multiple operations
     *
     * Creates a context that can handle multiple operations on a given ASIC.
     * This is useful when you need to perform several related operations
     * within the same context.
     *
     * @param[out] context Pointer to store the created context handle
     * @param asic String identifying the target ASIC/GPU architecture
     * @param opName First operation name
     * @param ... Variable argument list of additional operation names, terminated with NULL
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The variable argument list must be NULL-terminated
     * @example mlssCreateContextList(&ctx, "gfx1100", "gemm", "conv", "mha", NULL);
     */
    MLSS_API MLSSstatus mlssCreateContextList(MLSScontext* context, const MLSSstring asic, const MLSSstring opName, ...);

    // Parameters management

    /**
     * @brief Set a parameter value by name for a specific operation
     *
     * Sets the value of a named parameter for a specific operation within
     * the context. Parameters control various aspects of operation execution
     * such as dimensions, data types, and algorithm selection.
     *
     * @param context Pointer to the context (will be modified if needed)
     * @param opName Name of the operation to configure
     * @param parameterName Name of the parameter to set
     * @param value Pointer to the value to set (type depends on parameter)
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The value type must match the expected type for the parameter
     * @warning Incorrect value types may lead to undefined behavior
     */
    MLSS_API MLSSstatus mlssSetParameterByName(MLSScontext* const context, const MLSSstring opName, const MLSSstring parameterName, const MLSSvoid* const value);

    /**
     * @brief Set a parameter value by enumeration for a specific operation
     *
     * Sets the value of a parameter identified by an enumeration constant
     * for a specific operation within the context. This is typically faster
     * than setting by name as it avoids string comparisons.
     *
     * @param context Pointer to the context (will be modified if needed)
     * @param opName Name of the operation to configure
     * @param parameterFlag Enumeration value identifying the parameter
     * @param value Pointer to the value to set (type depends on parameter)
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The value type must match the expected type for the parameter
     */
    MLSS_API MLSSstatus mlssSetParameterByEnum(MLSScontext* const context, const MLSSstring opName, const MLSSenum parameterFlag, const MLSSvoid* const value);

    // get caps

    /**
     * @brief Get capability status for all operations in the context
     *
     * Queries the capabilities of all operations configured in the context
     * and returns an array of status codes indicating whether each operation
     * is supported with the current parameters.
     *
     * @param context The context to query capabilities for
     * @param[out] pStatuses Pointer to store the array of status codes
     * @param[out] nStatuses Pointer to store the number of statuses returned
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The caller is responsible for freeing the allocated status array
     * @warning The status array must be freed to avoid memory leaks
     */
    MLSS_API MLSSstatus mlssGetCaps(const MLSScontext context, MLSSstatus** const pStatuses, MLSSsize* const nStatuses);

    // Binary management

    /**
     * @brief Get shader binaries for all operations in the context
     *
     * Retrieves the compiled shader binaries for all operations configured
     * in the context. These binaries can be used for execution on the GPU.
     *
     * @param context The context to get binaries from
     * @param[out] binaries Pointer to store the array of binary structures
     * @param[out] numBinaries Pointer to store the number of binaries returned
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The caller is responsible for managing the returned binary array
     * @warning Binaries are specific to the ASIC and parameters configured
     */
    MLSS_API MLSSstatus mlssGetBinaries(const MLSScontext context, MLSSbinary** const binaries, MLSSsize* const numBinaries);

    // Operator management

    /**
     * @brief Enumerate all available operators in the AMDMLSS library
     *
     * Retrieves a list of all operator names supported by the library.
     * This can be used to discover available operations at runtime.
     *
     * @param[out] operators Array to store the operator names
     * @param[in,out] count On input, the size of the operators array; on output, the actual number of operators
     * @return MLSSvoid No return value
     * @note If operators is NULL, only the count is returned
     * @example To get the count: mlssEnumerateOperators(NULL, &count);
     */
    MLSS_API MLSSvoid mlssEnumerateOperators(MLSSstringarray operators, MLSSint32* count);

    // print functions

    /**
     * @brief Print all parameters for a specific operation
     *
     * Prints detailed information about all parameters configured for
     * a specific operation within the context. This is useful for
     * debugging and verification purposes.
     *
     * @param context The context containing the operation
     * @param opName Name of the operation to print parameters for
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note Output is printed to stdout
     */
    MLSS_API MLSSstatus mlssPrintParameters(const MLSScontext context, const MLSSstring opName);

    /**
     * @brief Print information about shader binaries
     *
     * Prints detailed information about an array of shader binaries,
     * including sizes, types, and other metadata. Useful for debugging
     * and analysis of generated shaders.
     *
     * @param binaries Array of binary structures to print
     * @param n Number of binaries in the array
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note Output is printed to stdout
     */
    MLSS_API MLSSstatus mlssPrintBinaries(const MLSSbinary* const binaries, const MLSSsize n);

    // vector management.

    /**
     * @brief Retrieve data from a vector object
     *
     * Extracts the raw data pointer, element count, and data type from
     * a vector object. This allows access to the underlying data for
     * processing or inspection.
     *
     * @param vector The vector object to retrieve data from
     * @param[out] data Pointer to store the data pointer (can be NULL)
     * @param[out] n Pointer to store the number of elements (can be NULL)
     * @param[out] type Pointer to store the data type enumeration (can be NULL)
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note Any of the output parameters can be NULL if that information is not needed
     */
    MLSS_API MLSSstatus mlssVectorRetrieveData(const MLSSvector vector,
                                               MLSSvoid** const data,
                                               MLSSsize* const n,
                                               MLSSenum* const type);

    /**
     * @brief Check if a vector object is valid
     *
     * Validates whether a vector object is properly initialized and
     * contains valid data. This can be used to verify vector integrity
     * before performing operations.
     *
     * @param vector The vector object to validate
     * @return MLSSbool TRUE if the vector is valid, FALSE otherwise
     */
    MLSS_API MLSSbool mlssVectorIsValid(const MLSSvector vector);

    // Verbose mode management

    /**
     * @brief Set the verbosity level for library output
     *
     * Controls the amount of diagnostic and informational output produced
     * by the library. Higher levels provide more detailed information.
     *
     * @param level The verbosity level to set (e.g., MLSS_VERBOSE_NONE, MLSS_VERBOSE_ERROR, MLSS_VERBOSE_INFO, MLSS_VERBOSE_DEBUG)
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note Default level is typically MLSS_VERBOSE_NONE
     * @see mlssGetVerboseLevel() to query the current level
     */
    MLSS_API MLSSstatus mlssSetVerboseLevel(MLSSenum level);

    /**
     * @brief Get the current verbosity level
     *
     * Returns the currently configured verbosity level for library output.
     *
     * @return MLSSenum The current verbosity level
     * @see mlssSetVerboseLevel() to change the level
     */
    MLSS_API MLSSenum mlssGetVerboseLevel();

    /**
     * @brief Enable verbose mode output
     *
     * Enables verbose output from the library. This is equivalent to
     * setting a non-zero verbose level.
     *
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note This may set the verbose level to a default value (e.g., MLSS_VERBOSE_INFO)
     * @see mlssDisableVerboseMode() to turn off verbose output
     */
    MLSS_API MLSSstatus mlssEnableVerboseMode();

    /**
     * @brief Disable verbose mode output
     *
     * Disables all verbose output from the library. This is equivalent to
     * setting the verbose level to MLSS_VERBOSE_NONE.
     *
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @see mlssEnableVerboseMode() to turn on verbose output
     */
    MLSS_API MLSSstatus mlssDisableVerboseMode();

    // Device query functions

    /**
     * @brief Get features and capabilities of all available devices
     *
     * Queries the system for all available GPU devices and retrieves their
     * features and capabilities, including compute units, memory sizes,
     * supported operations, and performance characteristics.
     *
     * @param[out] devices Pointer to store the array of device feature structures
     * @param[out] numDevices Pointer to store the number of devices found
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The caller is responsible for freeing the allocated device array
     * @warning Requires appropriate GPU drivers to be installed
     */
    MLSS_API MLSSstatus mlssGetDeviceFeatures(MLSSdevicefeatures** devices, MLSSsize* numDevices);

    /**
     * @brief Get features of the optimal device for AMDMLSS operations
     *
     * Automatically selects the best available GPU device for AMDMLSS
     * operations based on capabilities and performance characteristics,
     * and returns its features.
     *
     * @param[out] device Pointer to store the optimal device features
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note The selection criteria may include compute capability, memory size, and availability
     * @warning Returns an error if no suitable devices are found
     */
    MLSS_API MLSSstatus mlssGetOptimalDeviceFeatures(MLSSdevicefeatures* device);

    // Custom Type Registration Functions

    /**
     * @brief Register a custom type with comprehensive handlers
     *
     * Registers a custom type with its name and various handler functions. Once registered,
     * the custom type can be used with standard context operations by using the returned
     * type ID as the valueType parameter.
     *
     * @param typeName Name of the custom type (e.g., "MHAParameters", "ConvConfig")
     * @param setFunc Set function - sets parameters from custom type (required)
     * @param getFunc Get function - gets parameters into custom type (required)
     * @param ... Variable list of operator names this type can be used with, terminated with MLSS_END_LIST
     * @return MLSSenum Type ID that can be used as valueType in standard functions, 0 on failure
     * @note The type name must be unique. Registering with an existing name will fail.
     * @note setFunc and getFunc are required and cannot be NULL
     * @note The operator list must be terminated with MLSS_END_LIST
     * @example
     *   // Define handler functions for MHAParameters
     *   MLSSstatus setMHA(MLSScontext* ctx, const char* op, const void* data) {
     *       const MHAParameters* params = (const MHAParameters*)data;
     *       // Set parameters in context
     *       return MLSS_SUCCESS;
     *   }
     *
     *   MLSSstatus getMHA(MLSScontext ctx, const char* op, void** data) {
     *       MHAParameters* params = malloc(sizeof(MHAParameters));
     *       // Get parameters from context
     *       *data = params;
     *       return MLSS_SUCCESS;
     *   }
     *
     *   void printMHA(const void* data) {
     *       const MHAParameters* params = (const MHAParameters*)data;
     *       // Print MHA parameters
     *   }
     *
     *   // Register the custom type with variadic operator list
     *   MLSSenum MHA_TYPE = mlssRegisterCustomType("MHAParameters",
     *                                              setMHA, getMHA,
     *                                              MLSS_MHA, MLSS_GQA, MLSS_END_LIST);
     */
    MLSS_API MLSSenum mlssRegisterCustomType(const MLSSstring typeName,
                                             MLSScustomtypeset setFunc,
                                             MLSScustomtypeget getFunc,
                                             ...);

    /**
     * @brief Get information about a registered custom type
     *
     * Retrieves information about a custom type that was previously registered.
     *
     * @param customTypeId Type ID returned from mlssRegisterCustomType
     * @param[out] info Pointer to store the custom type information
     * @return MLSSstatus MLSS_STATUS_SUCCESS if type is registered,
     *                    MLSS_ERROR_PARAMETER_NOT_FOUND if type not found
     */
    MLSS_API MLSSstatus mlssGetCustomTypeInfo(MLSSenum customTypeId, MLSScustomtypeinfo* info);

    /**
     * @brief Unregister a custom type
     *
     * Removes a previously registered custom type from the system.
     * After unregistering, the type ID becomes invalid and cannot be used.
     *
     * @param customTypeId Type ID returned from mlssRegisterCustomType
     * @return MLSSstatus MLSS_STATUS_SUCCESS if unregistered successfully,
     *                    MLSS_ERROR_PARAMETER_NOT_FOUND if type not found
     * @note Any operations using this type will become invalid
     */
    MLSS_API MLSSstatus mlssUnregisterCustomType(MLSSenum customTypeId);

    /**
     * @brief Check if a type ID represents a custom type
     *
     * Determines whether a given type ID is a registered custom type.
     *
     * @param typeId Type ID to check
     * @return MLSSbool TRUE if typeId is a registered custom type, FALSE otherwise
     */
    MLSS_API MLSSbool mlssIsCustomType(MLSSenum typeId);

    /**
     * @brief Set a parameter value by name with explicit type information
     *
     * Sets the value of a named parameter for a specific operation within
     * the context with explicit type information. This is useful for custom
     * types where you want to pass pre-parsed objects directly.
     *
     * @param context Pointer to the context (will be modified if needed)
     * @param opName Name of the operation to configure
     * @param parameterName Name of the parameter to set
     * @param valueType The type of the value (including custom types)
     * @param value Pointer to the value to set
     * @return MLSSstatus MLSS_STATUS_SUCCESS on success, error code otherwise
     * @note For custom types, the value should be a pre-parsed object
     */
    MLSS_API MLSSstatus mlssSetParameterByNameTyped(MLSScontext* const context,
                                                    const MLSSstring opName,
                                                    const MLSSstring parameterName,
                                                    MLSSenum valueType,
                                                    const MLSSvoid* const value);

#ifdef __cplusplus
}
#endif
