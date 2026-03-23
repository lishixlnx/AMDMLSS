cmake_minimum_required(VERSION 3.25)
project(amdmlss VERSION 1.0.0 LANGUAGES C CXX)  # Added C language

# Load custom functions
include(cmake/utils.cmake)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Global settings for C++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global settings for C (for samples and API)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Enable comprehensive Visual Studio project structure
setup_vs_project_structure()

# Add standalone headers for better Visual Studio organization
add_standalone_headers(common_headers
    FOLDER "Libraries"
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/modules/common/include"
)

# Options
option(BUILD_SAMPLES "Build sample/test programs" OFF)
option(BUILD_TESTS "Build Catch2 unit tests" ON)

# Add Catch2 if tests are enabled and available
if(BUILD_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Catch2/CMakeLists.txt")
        add_subdirectory(3rdparty/Catch2)
    else()
        message(WARNING "Catch2 submodule missing; disabling BUILD_TESTS")
        set(BUILD_TESTS OFF CACHE BOOL "Build Catch2 unit tests" FORCE)
    endif()
endif()

# Add modules directory
add_subdirectory(modules)

# Add samples/tests directory if enabled
if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

# Add tests directory if enabled
if(BUILD_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        enable_testing()
        add_subdirectory(tests)
    else()
        message(WARNING "Tests directory missing; skipping test targets")
    endif()
endif()

# Installation settings
include(GNUInstallDirs)

install(EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

configure_package_config_file(
    cmake/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
    FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    DESTINATION 
        ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
