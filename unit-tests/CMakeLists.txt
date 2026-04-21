# GPU kernel integration tests for MLSS operators.
# Requires mlss-tester (installed) and the amdmlss C API.

set(_MLSS_TESTER_INSTALL "${CMAKE_SOURCE_DIR}/3rdparty/amd-mlss-tester/install")
find_package(mlss-tester REQUIRED
    PATHS "${_MLSS_TESTER_INSTALL}"
    PATH_SUFFIXES lib/cmake/mlss-tester
    NO_DEFAULT_PATH
)

if(NOT TARGET c_api)
    message(STATUS "c_api target not available; skipping GPU kernel tests")
    return()
endif()

include(CTest)

# Locate the ROCm HIP runtime DLL so the test can find hipLibrary* symbols
set(_runtime_dlls "")

if(MLSS_ENABLE_HIP AND hip_DIR)
    get_filename_component(_hip_root "${hip_DIR}/../../.." ABSOLUTE)
    find_file(_HIP_RUNTIME_DLL
        NAMES amdhip64_7.dll amdhip64.dll
        PATHS "${_hip_root}/bin"
        NO_DEFAULT_PATH
    )
    if(_HIP_RUNTIME_DLL)
        list(APPEND _runtime_dlls "${_HIP_RUNTIME_DLL}")
    endif()
endif()

list(REMOVE_DUPLICATES _runtime_dlls)

# Helper: create a GPU kernel test executable
function(add_gpu_kernel_test TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} "${SOURCE_FILE}")

    target_link_libraries(${TARGET_NAME} PRIVATE mlss::mlss-tester)

    # Only pull c_api's own headers + the common C defs; avoid transitive shaders include.
    target_link_libraries(${TARGET_NAME} PRIVATE $<TARGET_LINKER_FILE:c_api>)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common"
        "${CMAKE_SOURCE_DIR}/modules/c_api/include"
        "${CMAKE_BINARY_DIR}/modules/c_api/include"
        "${CMAKE_SOURCE_DIR}/modules/common/include"
    )

    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_23)

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /EHa)
    endif()

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    set_tests_properties(${TARGET_NAME} PROPERTIES SKIP_RETURN_CODE 77)

    # Copy the amdmlss DLL next to the test executable
    if(TARGET c_api)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:c_api>"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMENT "Copying amdmlss DLL for ${TARGET_NAME}"
        )
    endif()

    foreach(_dll IN LISTS _runtime_dlls)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMENT "Copying ${_dll} for ${TARGET_NAME}"
        )
    endforeach()

    set_target_properties(${TARGET_NAME} PROPERTIES
        FOLDER "Unit Tests"
    )
endfunction()

# MHA kernel test
add_gpu_kernel_test(mlss-mha-test "${CMAKE_CURRENT_LIST_DIR}/mha_test.cpp")

# GQA kernel test
add_gpu_kernel_test(mlss-gqa-test "${CMAKE_CURRENT_LIST_DIR}/gqa_test.cpp")
