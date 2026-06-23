# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# OpenCV-style fetch module for the amd-mlss-tester dependency.
#
# amd-mlss-tester is consumed only by the unit tests. It is obtained with
# FetchContent (when FETCH_TESTER is ON) or from a local checkout supplied via
# FETCHCONTENT_SOURCE_DIR_AMDMLSSTESTER (when FETCH_TESTER is OFF), then built
# in-tree with the same compiler and build options as the main project.
include_guard(GLOBAL)

include(FetchContent)
include("${CMAKE_SOURCE_DIR}/cmake/mlss_base64.cmake")

# Repository and revision are cache variables so CI or developers can pin a
# known-good commit/tag for reproducible builds (e.g.
# -DMLSS_TESTER_GIT_TAG=<sha>). The default tracks the moving 'main' branch (a
# branch name, not a fixed tag/commit), as the former submodule did. The default
# URL is stored Base64-encoded (decodes to the amd-mlss-tester repository URL).
mlss_base64_decode(_mlss_tester_default_repository
    "aHR0cHM6Ly9naXRodWIuY29tL0FNRC1SYWRlb24tTUwvYW1kLW1sc3MtdGVzdGVyLmdpdA==")
set(MLSS_TESTER_GIT_REPOSITORY "${_mlss_tester_default_repository}"
    CACHE STRING "amd-mlss-tester Git repository URL")
set(MLSS_TESTER_GIT_TAG "main"
    CACHE STRING "amd-mlss-tester Git tag, branch, or commit to fetch")

if(NOT FETCH_TESTER AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_AMDMLSSTESTER)
    message(FATAL_ERROR
        "FETCH_TESTER=OFF requires a local amd-mlss-tester checkout supplied via "
        "-DFETCHCONTENT_SOURCE_DIR_AMDMLSSTESTER=<path>.")
endif()

FetchContent_Declare(amdmlsstester
    GIT_REPOSITORY "${MLSS_TESTER_GIT_REPOSITORY}"
    GIT_TAG        "${MLSS_TESTER_GIT_TAG}"
    GIT_SHALLOW    TRUE
)

# Populate explicitly (rather than FetchContent_MakeAvailable) so the working
# tree can be inspected for the optional dx12 helper sources before the lib's
# CMakeLists is configured: the detection below sets cache variables the lib
# reads at configure time.
FetchContent_GetProperties(amdmlsstester)
if(NOT amdmlsstester_POPULATED)
    FetchContent_Populate(amdmlsstester)
endif()

if(NOT EXISTS "${amdmlsstester_SOURCE_DIR}/lib/CMakeLists.txt")
    message(FATAL_ERROR
        "amd-mlss-tester source is missing its lib/CMakeLists.txt at "
        "'${amdmlsstester_SOURCE_DIR}'.")
endif()

list(APPEND CMAKE_MODULE_PATH
    "${amdmlsstester_SOURCE_DIR}/cmake"
    "${amdmlsstester_SOURCE_DIR}/cmake/Modules")

set(MLSS_ENABLE_HIP ON CACHE BOOL "Enable HIP backend in mlss-tester" FORCE)

# D3D12 — Windows-only API. The backend additionally requires the
# amd-cross-compiler-tester helper sources (whose submodule path moved
# between mlss-tester revisions; check both legacy and current layouts).
if(WIN32)
    set(_DX12_SRC "")
    foreach(_candidate
            "${amdmlsstester_SOURCE_DIR}/3rdparty/amd-cross-compiler-tester/src"
            "${amdmlsstester_SOURCE_DIR}/amd-cross-compiler-tester/src")
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
        message(STATUS "mlss-tester: D3D disabled (amd-cross-compiler-tester sources not available)")
    endif()
else()
    set(MLSS_ENABLE_D3D OFF CACHE BOOL "D3D disabled (Windows-only backend)" FORCE)
    message(STATUS "mlss-tester: D3D disabled (Windows-only backend)")
endif()

# OpenCL — requires OpenCL SDK headers and library. If a system-wide
# SDK is not available, fall back to the tester's FetchOpenCL helper
# (it superseded cmake/Modules/FetchOpenCLSDK.cmake and now lives under
# 3rdparty/OpenCL). Including it runs fetch_opencl_sdk(), which downloads
# the Khronos pre-built SDK (Windows) or builds it from source (Linux) and
# exports OpenCL_ROOT / CMAKE_PREFIX_PATH so the next find_package succeeds.
find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    include("${amdmlsstester_SOURCE_DIR}/3rdparty/OpenCL/FetchOpenCL.cmake")
    find_package(OpenCL QUIET)
endif()
if(OpenCL_FOUND)
    set(MLSS_ENABLE_CL ON CACHE BOOL "Enable CL backend in mlss-tester" FORCE)
else()
    set(MLSS_ENABLE_CL OFF CACHE BOOL "CL disabled (OpenCL SDK not found)" FORCE)
    message(STATUS "mlss-tester: OpenCL disabled (SDK fetch failed)")
endif()

# AOCL — the tester's lib treats MLSS_ENABLE_AOCL=ON as a hard requirement
# (find_package(AOCL REQUIRED): BLIS + libFLAME), and there is no fetch helper
# for it. The unit tests only require the HIP, OpenCL, D3D (Windows) and Host
# backends (see unit-tests/CMakeLists.txt), not AOCL, so leave it disabled to
# avoid a mandatory system install of AOCL.
set(MLSS_ENABLE_AOCL OFF CACHE BOOL "AOCL not required by the unit tests; no fetch path available" FORCE)

# Build the tester as a STATIC library and stage it under 3rdparty as a
# self-contained, prebuilt dependency ("amd-mlss-tester-dep"). Static linking
# folds the tester into each unit-test executable, so no runtime DLL needs to
# travel alongside the tests.
set(_SAVE_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("${amdmlsstester_SOURCE_DIR}/lib"
                 "${CMAKE_BINARY_DIR}/3rdparty/mlss-tester")
set(BUILD_SHARED_LIBS "${_SAVE_BUILD_SHARED_LIBS}")

# Deploy the freshly built static library and its public headers into
# 3rdparty/amd-mlss-tester-dep/<config> so the artifact is available as a
# prebuilt dependency (gitignored; per-config to keep ABIs separate). A custom
# target is used (rather than add_custom_command(TARGET ...)) because the
# mlss-tester target is defined in the lib subdirectory scope, not here.
if(TARGET mlss-tester)
    set(_MLSS_TESTER_DEP_DIR "${CMAKE_SOURCE_DIR}/3rdparty/amd-mlss-tester-dep/$<CONFIG>")
    add_custom_target(deploy-mlss-tester ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_MLSS_TESTER_DEP_DIR}/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:mlss-tester>" "${_MLSS_TESTER_DEP_DIR}/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${amdmlsstester_SOURCE_DIR}/lib/include" "${_MLSS_TESTER_DEP_DIR}/include"
        DEPENDS mlss-tester
        COMMENT "Deploying mlss-tester static lib + headers to 3rdparty/amd-mlss-tester-dep"
        VERBATIM)
endif()
