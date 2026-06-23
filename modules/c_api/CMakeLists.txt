# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
add_project_module(c_api
    TYPE SHARED
    SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    HEADERS 
        "${CMAKE_CURRENT_SOURCE_DIR}/include/amdmlss/*.h"
    DEPENDENCIES
        common
        core
        shaders
)

# Link HIP runtime for device query functions
#find_and_add_hip(c_api)

# Set MLSS_FLAVOR based on build type
target_compile_definitions(c_api PRIVATE
    $<$<CONFIG:Debug>:MLSS_FLAVOR="dev">
    $<$<CONFIG:Release>:MLSS_FLAVOR="rel">
    $<$<CONFIG:RelWithDebInfo>:MLSS_FLAVOR="rel">
    $<$<CONFIG:MinSizeRel>:MLSS_FLAVOR="rel">
)

# Generate export header for the shared library
include(GenerateExportHeader)
generate_export_header(c_api
    BASE_NAME MLSS
    EXPORT_MACRO_NAME MLSS_API
    NO_EXPORT_MACRO_NAME MLSS_NO_EXPORT
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/amdmlss/amdmlss_export.h"
)

# Add the binary directory to include path so the generated header can be found
target_include_directories(c_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

# The public C-API headers (modules/c_api/include/amdmlss/amdmlss_api.h)
# include "amdmlss/amdmlss_api_cdefs.h" which lives in the header-only
# `common` module. Re-expose it as a PUBLIC dependency so consumers
# (samples, downstream apps) inherit its INTERFACE include path. `common`
# carries no symbols, so this does not leak implementation details.
target_link_libraries(c_api PUBLIC common)

# Set library properties
set_target_properties(c_api PROPERTIES
    OUTPUT_NAME "amdmlss"
    EXPORT_NAME amdmlss
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Install headers
install(
    DIRECTORY include/amdmlss
    DESTINATION include
    COMPONENT amdmlss
    FILES_MATCHING PATTERN "*.h"
)

# Install the generated export header
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/amdmlss/amdmlss_export.h"
    DESTINATION include/amdmlss
    COMPONENT amdmlss
)

# Install the library
install(TARGETS c_api
    EXPORT ${PROJECT_NAME}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT amdmlss
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
)
