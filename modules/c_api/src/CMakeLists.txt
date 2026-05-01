include(GenerateExportHeader)

# Collect all source files from src folder only
file(GLOB_RECURSE LIB_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

message("Sources: ${LIB_SOURCES}")

# Create SHARED library with OBJECT libraries for single output
add_library(amdmlss SHARED 
    ${LIB_SOURCES}
    $<TARGET_OBJECTS:amdmlss_core>      # Include core object files
    $<TARGET_OBJECTS:amdmlss_shaders>   # Include shader object files
)

# Generate the REAL export header (this replaces the temporary one)
generate_export_header(amdmlss
    BASE_NAME MLSS
    EXPORT_MACRO_NAME MLSS_API
    EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/amdmlss_export.h
    STATIC_DEFINE MLSS_STATIC_DEFINE
)

# Create alias for consistent naming
add_library(amdmlss::amdmlss ALIAS amdmlss)

# Include public and internal headers with updated paths for module prefixes
target_include_directories(amdmlss
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/modules                    # For "core/header.hpp" and "shaders/header.hpp"
        ${CMAKE_SOURCE_DIR}/modules/core/include       # Direct access to core headers  
        ${CMAKE_SOURCE_DIR}/modules/shaders/include    # Direct access to shader headers
        ${CMAKE_SOURCE_DIR}/include/internal
)

# Link libraries for dependencies (not the object files)
target_link_libraries(amdmlss
    PRIVATE 
        amdmlss::core
        amdmlss::shaders
)

# Set target properties
set_target_properties(amdmlss PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "amdmlss"
    EXPORT_NAME amdmlss
)

# Install ONLY the main target (single DLL/SO)
install(TARGETS amdmlss
    EXPORT amdmlssTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT amdmlss
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install main headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT amdmlss
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

# Install module headers with module prefixes
install(DIRECTORY ${CMAKE_SOURCE_DIR}/modules/core/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core
    COMPONENT amdmlss
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/modules/shaders/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shaders
    COMPONENT amdmlss
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

install(FILES ${CMAKE_BINARY_DIR}/include/amdmlss_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT amdmlss
)

# Export targets for the build tree
export(EXPORT amdmlssTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/amdmlssTargets.cmake
    NAMESPACE amdmlss::
)

# Install the export file
install(EXPORT amdmlssTargets
    FILE amdmlssTargets.cmake
    NAMESPACE amdmlss::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/amdmlss
    COMPONENT amdmlss
)
