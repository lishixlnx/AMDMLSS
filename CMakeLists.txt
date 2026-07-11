# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Build every target as position-independent code. Required on Linux/ELF
# because the project's intermediate libraries (core, shaders, ...) are
# static archives that get linked into a shared library (libamdmlss.so);
# without -fPIC the linker rejects the relocations. MSVC silently ignores
# this flag, so Windows builds are unaffected.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    option(USE_BINSKIM_COMPLIANT_COMPILE_FLAGS "Prepare AMDMLSS for BinSkim Binary Analyzer" OFF)
    option(USE_SPECTRE_MITIGATED_LIBRARIES "Use Spectre-mitigated libraries" OFF)
    if(MSVC AND MSVC_VERSION GREATER_EQUAL 1900)
        if(USE_BINSKIM_COMPLIANT_COMPILE_FLAGS)
            message(STATUS "Control Flow Guard enabled")
            add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/guard:cf>)
            add_link_options(/guard:cf /DYNAMICBASE)
        endif()
        if(USE_SPECTRE_MITIGATED_LIBRARIES)
            message(STATUS "Using Spectre-mitigated libraries")
            add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Qspectre>)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(USE_BINSKIM_COMPLIANT_COMPILE_FLAGS)
            message(STATUS "Control Flow Guard enabled (Clang)")
            add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:-Xclang -cfguard>")
            add_link_options(-Wl,-guard:cf)
        endif()
        if(USE_SPECTRE_MITIGATED_LIBRARIES)
            message(WARNING "Spectre-mitigated libraries are not yet available for Clang in this project.")
        endif()
    endif()
    option(USE_MSVC_STATIC_RUNTIME "Link MSVC runtime library statically instead of dynamically" OFF)
    if(NOT USE_MSVC_STATIC_RUNTIME)
        set(MSVC_RUNTIME_SUFFIX "DLL")
    endif()
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>${MSVC_RUNTIME_SUFFIX}")
    endif()
endif()

# Enable comprehensive Visual Studio project structure
setup_vs_project_structure()

# Add standalone headers for better Visual Studio organization
add_standalone_headers(common_headers
    FOLDER "Libraries"
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/modules/common/include"
)

# Options
option(BUILD_UNIT_TESTS   "Build Catch2 GPU-kernel unit tests" ON)
option(BUILD_SAMPLE_TESTS "Build sample/test programs"         ON)

# The test dependencies (Catch2, amd-mlss-tester) are consumed only by the unit
# tests, so their fetch toggles default to the unit-tests flag. Set either to
# OFF to build the unit tests against a local/vendored copy without network
# access (supply the location via FETCHCONTENT_SOURCE_DIR_<NAME>).
option(FETCH_CATCH2 "Download Catch2 via FetchContent"          ${BUILD_UNIT_TESTS})
option(FETCH_TESTER "Download amd-mlss-tester via FetchContent" ${BUILD_UNIT_TESTS})

# Add modules directory
add_subdirectory(modules)

# Add samples/tests directory if enabled
if(BUILD_SAMPLE_TESTS)
    enable_testing()
    add_subdirectory(samples)
endif()

# Add tests directory if enabled
if(BUILD_UNIT_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        enable_testing()
        add_subdirectory(tests)
    else()
        message(WARNING "Tests directory missing; skipping test targets")
    endif()
endif()

# Provide hip::host so the mlss-tester driver subdirectory and the unit-tests
# subdirectory can link HIP headers + runtime. Both consumers are gated on
# BUILD_UNIT_TESTS below, so there is no point trying to import the target
# outside that mode. We prefer ROCm's own hip-config.cmake (it imports the full
# export set hip::host / hip::amdhip64 / hip::device); pre-creating only
# hip::host manually trips the export-set safety check inside hip-targets.cmake
# and breaks downstream find_package(hip), so we do not fall back to a manual stub.
if(BUILD_UNIT_TESTS AND NOT TARGET hip::host)
    # Force CONFIG mode: ROCm ships a module-mode Findhip.cmake on
    # CMAKE_MODULE_PATH (C:/opt/rocm/cmake) that only sets HIP_* variables and
    # never creates the hip::host imported target. The hip::host / hip::device
    # export set lives exclusively in the config package (hip-config.cmake), so
    # config mode is required for the mlss-tester and unit-test consumers.
    find_package(hip CONFIG QUIET)
endif()

# Test dependencies are consumed only by the unit tests. Each is encapsulated in
# a self-contained fetch module under 3rdparty/<dep>/<dep>.cmake (OpenCV style)
# and pulled in only when the unit tests are built, so library-only and
# samples-only builds never trigger a fetch.
if(BUILD_UNIT_TESTS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Catch2/Catch2.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/amd-mlss-tester/amd-mlss-tester.cmake)
else()
    message(STATUS "BUILD_UNIT_TESTS=OFF — skipping Catch2 / amd-mlss-tester")
endif()

# Add GPU kernel unit tests
if(BUILD_UNIT_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit-tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(unit-tests)
endif()

# Installation settings
include(GNUInstallDirs)

install(EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    COMPONENT amdmlss
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
    COMPONENT amdmlss
)

install(
    FILES cmake/FindHIP.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/Modules
    COMPONENT amdmlss
)
