# GPU kernel integration tests for MLSS operators.
# Requires the mlss-tester library target and the amdmlss C API.

if(NOT TARGET mlss-tester)
    message(STATUS "mlss-tester target not available; skipping GPU kernel tests")
    return()
endif()

if(NOT TARGET c_api)
    message(STATUS "c_api target not available; skipping GPU kernel tests")
    return()
endif()

# ── Required mlss-tester backends ────────────────────────────────────────────
# The unit tests exercise every backend the tester supports. HIP and OpenCL
# are required on every platform; D3D12 is Windows-only and therefore only
# required when building for Windows. The Host tensor library (used for
# reference computation) is always compiled into mlss-tester.
set(_MLSS_REQUIRED_BACKENDS HIP CL)
if(WIN32)
    list(APPEND _MLSS_REQUIRED_BACKENDS D3D)
endif()
foreach(_backend IN LISTS _MLSS_REQUIRED_BACKENDS)
    if(NOT MLSS_ENABLE_${_backend})
        message(FATAL_ERROR
            "unit-tests: mlss-tester backend ${_backend} is not enabled "
            "(MLSS_ENABLE_${_backend}=OFF). The unit tests require HIP and "
            "OpenCL on every platform (plus D3D12 on Windows), and the "
            "always-compiled Host tensor library.")
    endif()
endforeach()
if(WIN32)
    message(STATUS "unit-tests: mlss-tester backends enabled — HIP, D3D, OpenCL, Host")
else()
    message(STATUS "unit-tests: mlss-tester backends enabled — HIP, OpenCL, Host (D3D is Windows-only)")
endif()

include(CTest)

# Fetch Catch2 if not already available
if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()

# Locate the ROCm HIP runtime DLL so the test can find hipLibrary* symbols
set(_runtime_dlls "")

if(HIP_DIR)
    find_file(_HIP_RUNTIME_DLL
        NAMES amdhip64_7.dll amdhip64.dll
        PATHS "${HIP_DIR}/bin"
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

    target_link_libraries(${TARGET_NAME} PRIVATE
        mlss-tester
        Catch2::Catch2WithMain
    )

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
    set_tests_properties(${TARGET_NAME} PROPERTIES SKIP_RETURN_CODE 4)

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

    # Copy the mlss-tester DLL next to the test executable
    if(TARGET mlss-tester AND WIN32)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:mlss-tester>"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMENT "Copying mlss-tester DLL for ${TARGET_NAME}"
        )
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        FOLDER "Unit Tests"
    )
endfunction()

# MHA kernel test
add_gpu_kernel_test(mlss-mha-test "${CMAKE_CURRENT_LIST_DIR}/mha_test.cpp")

# GQA kernel test
add_gpu_kernel_test(mlss-gqa-test "${CMAKE_CURRENT_LIST_DIR}/gqa_test.cpp")

# Conv 1x1 kernel test
add_gpu_kernel_test(mlss-convmxn-test "${CMAKE_CURRENT_LIST_DIR}/convmxn_test.cpp")

# MVN2 (InstaNorm) kernel test
add_gpu_kernel_test(mlss-mvn2-test "${CMAKE_CURRENT_LIST_DIR}/mvn2_test.cpp")

# GEMM MxN (HIP + CK) kernel test
add_gpu_kernel_test(mlss-gemm-test "${CMAKE_CURRENT_LIST_DIR}/gemm_test.cpp")
