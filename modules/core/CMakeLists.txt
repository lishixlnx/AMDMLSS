# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
add_project_module(core
    TYPE STATIC
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/core/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/core/*.hxx"
    DEPENDENCIES
        common
)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

if(NOT DEFINED HIP_PATH OR "${HIP_PATH}" STREQUAL "")
    if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        set(HIP_PATH "$ENV{HIP_PATH}")
    elseif(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(HIP_PATH "$ENV{ROCM_PATH}")
    else()
        message(FATAL_ERROR "HIP_PATH or ROCM_PATH environment variable must be defined before configuring the core module")
    endif()
endif()

if(NOT EXISTS "${HIP_PATH}")
    message(FATAL_ERROR "HIP_PATH points to a non-existent directory: ${HIP_PATH}")
endif()

# Share the resolved path with the custom FindHIP module
set(HIP_DIR "${HIP_PATH}" CACHE PATH "Location of the HIP installation" FORCE)

find_package(HIP REQUIRED)

target_link_libraries(core PRIVATE HIP::Runtime)

find_package(amd_comgr REQUIRED
    PATHS "${HIP_PATH}" "$ENV{ROCM_PATH}"
    PATH_SUFFIXES lib/cmake/amd_comgr
    NO_DEFAULT_PATH
)

target_link_libraries(core PRIVATE amd_comgr)

message(STATUS "HIP found using FindHIP.cmake at: ${HIP_DIR}")
message(STATUS "AMD Comgr found via find_package(amd_comgr)")

if(WIN32)
    find_package(WindowsSDK REQUIRED COMPONENTS DXGI D3D12)
    target_link_libraries(core PRIVATE WindowsSDK::DXGI WindowsSDK::D3D12)
    if(WindowsSDK_DXGI_HEADER)
        target_include_directories(core PRIVATE "${WindowsSDK_DXGI_HEADER}")
    endif()
    message(STATUS "Windows SDK found using FindWindowsSDK.cmake")

    if(NOT MLSS_USE_RUNTIME)
        find_library(WindowsSDK_DXCompiler_LIBRARY
            NAMES dxcompiler
            PATHS "${WIN10_SDK_LIBRARY_DIR}"
            NO_DEFAULT_PATH)

        if(NOT WindowsSDK_DXCompiler_LIBRARY)
            message(FATAL_ERROR "dxcompiler library not found in Windows SDK; required when MLSS_USE_RUNTIME=0")
        endif()

        target_link_libraries(core PRIVATE ${WindowsSDK_DXCompiler_LIBRARY})
    endif()

    set(_mlss_runtime_dlls)
    set(_win_sdk_bin_dir "${WIN10_SDK_PATH}/bin/${WIN10_SDK_VERSION}/x64")

    find_file(WindowsSDK_DXCompiler_DLL
        NAMES dxcompiler.dll
        PATHS
            "${_win_sdk_bin_dir}"
            "$ENV{SystemRoot}/System32"
            "$ENV{SystemRoot}/SysWOW64"
        PATH_SUFFIXES ""
    )

    find_file(WindowsSDK_DXGI_DLL
        NAMES dxgi.dll
        PATHS
            "$ENV{SystemRoot}/System32"
            "$ENV{SystemRoot}/SysWOW64"
    )

    find_file(WindowsSDK_D3D12_DLL
        NAMES d3d12.dll
        PATHS
            "$ENV{SystemRoot}/System32"
            "$ENV{SystemRoot}/SysWOW64"
    )

    find_file(HIP_RUNTIME_DLL
        NAMES amdhip64.dll
        PATHS
            "${HIP_DIR}/bin"
            "${HIP_DIR}/lib"
        NO_DEFAULT_PATH
    )

    if(NOT HIP_RUNTIME_DLL OR "${HIP_RUNTIME_DLL}" STREQUAL "HIP_RUNTIME_DLL-NOTFOUND")
        file(GLOB _hip_runtime_candidates "${HIP_DIR}/bin/amdhip64*.dll")
        if(_hip_runtime_candidates)
            list(SORT _hip_runtime_candidates COMPARE NATURAL ORDER DESCENDING)
            list(GET _hip_runtime_candidates 0 HIP_RUNTIME_DLL)
        endif()
    endif()

    foreach(_candidate IN ITEMS WindowsSDK_DXCompiler_DLL WindowsSDK_DXGI_DLL WindowsSDK_D3D12_DLL HIP_RUNTIME_DLL)
        if(${_candidate} AND NOT "${${_candidate}}" STREQUAL "${_candidate}-NOTFOUND")
            list(APPEND _mlss_runtime_dlls "${${_candidate}}")
            message(STATUS "Found ${_candidate}: ${${_candidate}}")
        else()
            message(WARNING "${_candidate} not found")
        endif()
    endforeach()

    if(_mlss_runtime_dlls)
        set(MLSS_RUNTIME_DLLS "${_mlss_runtime_dlls}" CACHE INTERNAL "Runtime DLLs required by MLSS")
        message(STATUS "MLSS runtime DLLs: ${MLSS_RUNTIME_DLLS}")
    endif()
endif()

# Install the target to export set
install(TARGETS core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
)
