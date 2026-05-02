# Create header-only interface library
add_library(common INTERFACE)

# Set C++23 for consumers
target_compile_features(common INTERFACE cxx_std_23)

# Add include directories
target_include_directories(common
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Install headers
install(
    DIRECTORY include/amdmlss
    DESTINATION include
    COMPONENT amdmlss
    FILES_MATCHING PATTERN "*.h"
)

# Install the target to export set
install(TARGETS common
    EXPORT ${PROJECT_NAME}-targets
    COMPONENT amdmlss
)
