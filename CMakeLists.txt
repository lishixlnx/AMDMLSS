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

# Use a vendored Catch2 checkout when available; otherwise unit-tests/CMakeLists.txt
# falls back to FetchContent. This keeps builds working both offline (with the
# submodule populated) and on fresh clones (network fetch).
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Catch2/CMakeLists.txt")
    add_subdirectory(3rdparty/Catch2)
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

# Provide hip::host so mlss-tester (and unit-tests) can link HIP headers + runtime
if(NOT TARGET hip::host)
    find_path(_MLSS_HIP_INC NAMES hip/hip_runtime.h
        PATH_SUFFIXES include PATHS "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}" NO_DEFAULT_PATH)
    find_library(_MLSS_HIP_LIB NAMES amdhip64
        PATH_SUFFIXES lib PATHS "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}" NO_DEFAULT_PATH)
    if(_MLSS_HIP_INC AND _MLSS_HIP_LIB)
        add_library(hip::host IMPORTED SHARED GLOBAL)
        set_target_properties(hip::host PROPERTIES
            IMPORTED_LOCATION "${_MLSS_HIP_LIB}"
            IMPORTED_IMPLIB "${_MLSS_HIP_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${_MLSS_HIP_INC}"
            INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__)
    endif()
endif()

# Build mlss-tester library from the submodule (same compiler and build options as the main project)
set(_MLSS_TESTER_SUBMODULE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/amd-mlss-tester")
ensure_git_submodule("3rdparty/amd-mlss-tester" "lib/CMakeLists.txt")
if(EXISTS "${_MLSS_TESTER_SUBMODULE}/lib/CMakeLists.txt")
    list(APPEND CMAKE_MODULE_PATH "${_MLSS_TESTER_SUBMODULE}/cmake/Modules")

    set(MLSS_ENABLE_HIP ON CACHE BOOL "Enable HIP backend in mlss-tester" FORCE)

    # D3D12 — requires the amd-cross-compiler-tester helper sources.
    # The submodule path moved between mlss-tester revisions; check both.
    set(_DX12_SRC "")
    foreach(_candidate
            "${_MLSS_TESTER_SUBMODULE}/3rdparty/amd-cross-compiler-tester/src"
            "${_MLSS_TESTER_SUBMODULE}/amd-cross-compiler-tester/src")
        if(EXISTS "${_candidate}/dx12")
            set(_DX12_SRC "${_candidate}")
            break()
        endif()
    endforeach()
    if(_DX12_SRC)
        set(MLSS_ENABLE_D3D ON CACHE BOOL "Enable D3D backend in mlss-tester" FORCE)
        set(MLSS_DX12_INCLUDE_DIR "${_DX12_SRC}"
            CACHE PATH "Path to the dx12 helper headers" FORCE)
    else()
        set(MLSS_ENABLE_D3D OFF CACHE BOOL "D3D disabled (amd-cross-compiler-tester not available)" FORCE)
        message(STATUS "mlss-tester: D3D disabled (amd-cross-compiler-tester submodule not initialised)")
    endif()

    # OpenCL — requires OpenCL SDK headers and library. If a system-wide
    # SDK is not available, fall back to the tester's FetchOpenCLSDK helper
    # which downloads the Khronos pre-built SDK (Windows) or builds it from
    # source (Linux). This guarantees the OpenCL backend is always compiled.
    find_package(OpenCL QUIET)
    if(NOT OpenCL_FOUND)
        include(FetchOpenCLSDK)
        fetch_opencl_sdk()
        find_package(OpenCL QUIET)
    endif()
    if(OpenCL_FOUND)
        set(MLSS_ENABLE_CL ON CACHE BOOL "Enable CL backend in mlss-tester" FORCE)
    else()
        set(MLSS_ENABLE_CL OFF CACHE BOOL "CL disabled (OpenCL SDK not found)" FORCE)
        message(STATUS "mlss-tester: OpenCL disabled (SDK fetch failed)")
    endif()

    # AOCL — probed quietly inside the lib, safe to always request
    set(MLSS_ENABLE_AOCL ON CACHE BOOL "Enable AOCL in mlss-tester" FORCE)

    set(_SAVE_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ON)
    add_subdirectory("${_MLSS_TESTER_SUBMODULE}/lib"
                     "${CMAKE_BINARY_DIR}/3rdparty/mlss-tester")
    set(BUILD_SHARED_LIBS "${_SAVE_BUILD_SHARED_LIBS}")
endif()

# Add GPU kernel unit tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit-tests/CMakeLists.txt")
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
