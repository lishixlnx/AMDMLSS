#cmake_minimum_required(VERSION 3.24)
cmake_minimum_required(VERSION 3.13.4)

if(NOT WIN32)
    message(FATAL_ERROR "Platform not supported")
endif()

include(FindPackageHandleStandardArgs)

set(WIN10_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")

# Resolve the SDK version from multiple sources so the finder works for both
# Visual Studio and Ninja/Clang generators.
set(_win10_sdk_version "${WIN10_SDK_VERSION}")

if(_win10_sdk_version STREQUAL "" AND DEFINED CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION AND NOT "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}" STREQUAL "")
    set(_win10_sdk_version "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
elseif(_win10_sdk_version STREQUAL "" AND DEFINED ENV{CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION} AND NOT "$ENV{CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}" STREQUAL "")
    set(_win10_sdk_version "$ENV{CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
elseif(_win10_sdk_version STREQUAL "" AND DEFINED ENV{WindowsSDKVersion} AND NOT "$ENV{WindowsSDKVersion}" STREQUAL "")
    set(_win10_sdk_version "$ENV{WindowsSDKVersion}")
elseif(_win10_sdk_version STREQUAL "" AND DEFINED ENV{WIN10_SDK_VERSION} AND NOT "$ENV{WIN10_SDK_VERSION}" STREQUAL "")
    set(_win10_sdk_version "$ENV{WIN10_SDK_VERSION}")
endif()

if(_win10_sdk_version STREQUAL "")
    file(GLOB _win10_sdk_candidates RELATIVE "${WIN10_SDK_PATH}/Include" "${WIN10_SDK_PATH}/Include/*")
    set(_win10_sdk_versions "")
    foreach(_candidate IN LISTS _win10_sdk_candidates)
        if(IS_DIRECTORY "${WIN10_SDK_PATH}/Include/${_candidate}" AND _candidate MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
            list(APPEND _win10_sdk_versions "${_candidate}")
        endif()
    endforeach()
    if(_win10_sdk_versions)
        list(SORT _win10_sdk_versions COMPARE NATURAL ORDER DESCENDING)
        list(GET _win10_sdk_versions 0 _win10_sdk_version)
        message(STATUS "Auto-detected Windows SDK version: ${_win10_sdk_version}")
    endif()
    unset(_win10_sdk_versions)
    unset(_win10_sdk_candidates)
endif()

set(WIN10_SDK_VERSION "${_win10_sdk_version}" CACHE STRING "Windows 10 SDK version" FORCE)
unset(_win10_sdk_version)

set(WIN10_SDK_LIBRARY_DIR "${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}")
set(WIN10_SDK_INCLUDE_DIR "${WIN10_SDK_PATH}/Include/${WIN10_SDK_VERSION}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(WIN10_SDK_LIBRARY_DIR "${WIN10_SDK_LIBRARY_DIR}/um/x64")
else()
    set(WIN10_SDK_LIBRARY_DIR "${WIN10_SDK_LIBRARY_DIR}/um/x86")
endif()

find_library(WindowsSDK_D3D12_LIBRARY 
    NAMES 
        d3d12 
    PATHS 
        ${WIN10_SDK_LIBRARY_DIR})

find_library(WindowsSDK_DXGI_LIBRARY 
    NAMES 
        dxgi
    PATHS 
        ${WIN10_SDK_LIBRARY_DIR})

find_path(WindowsSDK_DXGI_HEADER
    NAMES 
        dxgi.h
    PATH_SUFFIXES 
        shared
    PATHS 
        ${WIN10_SDK_INCLUDE_DIR})

if(WindowsSDK_D3D12_LIBRARY)
    set(WindowsSDK_D3D12_FOUND ON)
endif()

if(WindowsSDK_DXGI_LIBRARY AND WindowsSDK_DXGI_HEADER)
    set(WindowsSDK_DXGI_FOUND ON)
endif()

find_package_handle_standard_args(WindowsSDK HANDLE_COMPONENTS
    REASON_FAILURE_MESSAGE "Please install Windows SDK using Visual Studio Installer")

if(WindowsSDK_D3D12_FOUND)
    mark_as_advanced(WindowsSDK_D3D12_LIBRARY)
endif()

if(WindowsSDK_DXGI_FOUND)
    mark_as_advanced(WindowsSDK_DXGI_LIBRARY WindowsSDK_DXGI_HEADER)
endif()

if(WindowsSDK_DXGI_FOUND AND NOT TARGET WindowsSDK::D3D12)
    add_library(WindowsSDK::D3D12 STATIC IMPORTED)
    set_property(TARGET WindowsSDK::D3D12 PROPERTY 
        IMPORTED_LOCATION ${WindowsSDK_D3D12_LIBRARY})
endif()

if(WindowsSDK_DXGI_FOUND AND NOT TARGET WindowsSDK::DXGI)
    add_library(WindowsSDK::DXGI STATIC IMPORTED)
    set_target_properties(WindowsSDK::DXGI PROPERTIES 
        IMPORTED_LOCATION ${WindowsSDK_DXGI_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${WindowsSDK_DXGI_HEADER})
endif()
