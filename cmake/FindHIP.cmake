cmake_minimum_required(VERSION 3.24)
include(FindPackageHandleStandardArgs)



find_path(HIP_INCLUDE_DIR 
    NAMES 
        hip/hip_runtime.h
    PATH_SUFFIXES 
        include
    PATHS 
        $ENV{HIP_PATH}
        $ENV{ROCM_PATH}
    NO_DEFAULT_PATH
)

find_library(HIP_LIBRARY 
    NAMES 
        amdhip64
    PATH_SUFFIXES 
        lib
    PATHS 
        $ENV{HIP_PATH}
        $ENV{ROCM_PATH}
    NO_DEFAULT_PATH
)

find_package_handle_standard_args(HIP
    REQUIRED_VARS 
        HIP_INCLUDE_DIR 
        HIP_LIBRARY
    REASON_FAILURE_MESSAGE "Please install AMD HIP SDK")

if(HIP_FOUND)
    mark_as_advanced(HIP_LIBRARY HIP_INCLUDE_DIR)
endif()

if(HIP_FOUND AND NOT TARGET HIP::Runtime)
    add_library(HIP::Runtime IMPORTED STATIC)
    set_target_properties(HIP::Runtime PROPERTIES 
        IMPORTED_LOCATION ${HIP_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${HIP_INCLUDE_DIR}
        INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__)
endif()
