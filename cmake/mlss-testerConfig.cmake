# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# find_package(mlss-tester CONFIG).
# Same-build: the `mlss-tester` target already exists (add_subdirectory(lib) ran first).
# Installed layout: load exported targets from install(EXPORT) (not export(TARGETS)).
if(TARGET mlss-tester)
    if(NOT TARGET mlss::mlss-tester)
        add_library(mlss::mlss-tester ALIAS mlss-tester)
    endif()
else()
    include("${CMAKE_CURRENT_LIST_DIR}/mlss-tester-targets.cmake")
endif()
