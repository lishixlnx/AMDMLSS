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
    # If the ROCm config-file package was already loaded (hip::amdhip64 exists),
    # create HIP::Runtime as a thin wrapper that exposes __HIP_PLATFORM_AMD__
    # and delegates linking to hip::amdhip64, avoiding the ninja "no rule to
    # make hip::host;hip::device" error from the SHARED imported targets.
    if(TARGET hip::amdhip64)
        add_library(HIP::Runtime INTERFACE IMPORTED GLOBAL)
        set_target_properties(HIP::Runtime PROPERTIES
            INTERFACE_LINK_LIBRARIES "hip::amdhip64"
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
            INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__"
        )
    else()
        add_library(HIP::Runtime IMPORTED SHARED)
        set_target_properties(HIP::Runtime PROPERTIES
            IMPORTED_LOCATION "${HIP_LIBRARY}"
            IMPORTED_IMPLIB "${HIP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
            INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__"
        )
    endif()
endif()
