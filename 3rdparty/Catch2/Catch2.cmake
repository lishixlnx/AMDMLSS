# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Catch2 is consumed only by the unit tests. It is obtained with FetchContent
# (when FETCH_CATCH2 is ON) or located on the system / via a local copy (when
# FETCH_CATCH2 is OFF).
include_guard(GLOBAL)

if(TARGET Catch2::Catch2WithMain)
    return()
endif()

if(FETCH_CATCH2)
    include(FetchContent)
    include("${CMAKE_SOURCE_DIR}/cmake/mlss_base64.cmake")
    # Repository and revision are cache variables so CI or developers can pin a
    # known-good tag/commit for reproducible builds (e.g.
    # -DCATCH2_GIT_TAG=v3.7.1) instead of the default, which tracks the moving
    # 'devel' branch (a branch name, not a fixed tag/commit). The default
    # repository URL is stored Base64-encoded (upstream Catch2 on GitHub).
    mlss_base64_decode(_catch2_default_repository
        "aHR0cHM6Ly9naXRodWIuY29tL2NhdGNob3JnL0NhdGNoMi5naXQ=")
    set(CATCH2_GIT_REPOSITORY "${_catch2_default_repository}"
        CACHE STRING "Catch2 Git repository URL")
    set(CATCH2_GIT_TAG "devel"
        CACHE STRING "Catch2 Git tag, branch, or commit to fetch")
    FetchContent_Declare(Catch2
        GIT_REPOSITORY "${CATCH2_GIT_REPOSITORY}"
        GIT_TAG        "${CATCH2_GIT_TAG}"
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
else()
    find_package(Catch2 3 REQUIRED)
    if(DEFINED Catch2_SOURCE_DIR)
        list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
    endif()
endif()
