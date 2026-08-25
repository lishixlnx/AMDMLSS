# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Get all test files
file(GLOB TEST_SOURCES "*.c")

# Add each test file as a separate executable
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the base name without extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create test
    add_project_test(${TEST_NAME}
        ${TEST_SOURCE}
        DEPENDENCIES c_api
    )
endforeach()

# HIP module-load validation for Conv MxN (MIGraphX-style path on gfx1151 hardware).
set(_CONVMXN_HIP_VALIDATE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/convmxn_hip_module_validate.cpp")
if(EXISTS "${_CONVMXN_HIP_VALIDATE_SOURCE}")
    if(NOT TARGET hip::host)
        find_package(hip CONFIG QUIET)
    endif()

    if(TARGET hip::host)
        add_project_test(convmxn_hip_module_validate
            ${_CONVMXN_HIP_VALIDATE_SOURCE}
            DEPENDENCIES c_api
        )
        target_link_libraries(convmxn_hip_module_validate PRIVATE hip::host)

        if(HIP_DIR)
            find_file(_HIP_RUNTIME_DLL
                NAMES amdhip64_7.dll amdhip64.dll
                PATHS "${HIP_DIR}/bin"
                NO_DEFAULT_PATH
            )
            if(_HIP_RUNTIME_DLL)
                add_custom_command(TARGET convmxn_hip_module_validate POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_HIP_RUNTIME_DLL}"
                        "$<TARGET_FILE_DIR:convmxn_hip_module_validate>"
                    COMMENT "Copying HIP runtime DLL for convmxn_hip_module_validate"
                )
            endif()
        endif()
    else()
        message(STATUS "samples: skipping convmxn_hip_module_validate (hip::host not found)")
    endif()
endif()

# Static analysis of embedded Winograd Base gfx1100/fp16 archives.
set(_WINOGRAD_FP16_ANALYZE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/winograd_fp16_gfx1100_bin_analyze.cpp")
if(EXISTS "${_WINOGRAD_FP16_ANALYZE_SOURCE}")
    if(NOT TARGET hip::host)
        find_package(hip CONFIG QUIET)
    endif()

    if(TARGET hip::host)
        add_project_test(winograd_fp16_gfx1100_bin_analyze
            ${_WINOGRAD_FP16_ANALYZE_SOURCE}
        )
        target_link_libraries(winograd_fp16_gfx1100_bin_analyze PRIVATE hip::host)
        target_include_directories(winograd_fp16_gfx1100_bin_analyze PRIVATE
            "${CMAKE_SOURCE_DIR}/modules/shaders/include"
            "${CMAKE_SOURCE_DIR}/modules/core/include"
            "${CMAKE_SOURCE_DIR}/modules/common/include"
            "${CMAKE_SOURCE_DIR}/modules/shaders/src/operators/impl/conv/mxn/Winograd/Base"
        )

        if(HIP_DIR)
            find_file(_HIP_RUNTIME_DLL_ANALYZE
                NAMES amdhip64_7.dll amdhip64.dll
                PATHS "${HIP_DIR}/bin"
                NO_DEFAULT_PATH
            )
            if(_HIP_RUNTIME_DLL_ANALYZE)
                add_custom_command(TARGET winograd_fp16_gfx1100_bin_analyze POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_HIP_RUNTIME_DLL_ANALYZE}"
                        "$<TARGET_FILE_DIR:winograd_fp16_gfx1100_bin_analyze>"
                    COMMENT "Copying HIP runtime DLL for winograd_fp16_gfx1100_bin_analyze"
                )
            endif()
        endif()
    else()
        message(STATUS "samples: skipping winograd_fp16_gfx1100_bin_analyze (hip::host not found)")
    endif()
endif()
