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
